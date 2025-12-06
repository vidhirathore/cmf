import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import os
import pandas as pd

# Create directory for results
os.makedirs("results", exist_ok=True)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. The Causal Environment (SCM)
# ==========================================
class CausalManifoldSCM:
    def sample(self, n_samples=1000):
        # Sensitive Attribute: Binary
        A = torch.randint(0, 2, (n_samples, 1)).float()
        
        # Intrinsic Latent: Uniform manifold coordinate
        U = torch.rand(n_samples, 1) * 4 * np.pi
        
        # Structural Equation for X (The Warping)
        # A changes the frequency and radius of the roll
        w = 1.0 + 0.5 * A 
        x1 = (U * w) * torch.cos(U)
        x2 = (U * w) * torch.sin(U)
        x3 = U + (A * 2.0) 
        X = torch.cat([x1, x2, x3], dim=1)
        
        # Target Y: Depends on U (fair) 
        Y = (torch.sin(U) > 0).float()
        
        return X, A, Y, U

    def get_counterfactual(self, X, A, U, target_a_val):
        A_cf = torch.ones_like(A) * target_a_val
        w = 1.0 + 0.5 * A_cf
        x1 = (U * w) * torch.cos(U)
        x2 = (U * w) * torch.sin(U)
        x3 = U + (A_cf * 2.0)
        X_cf = torch.cat([x1, x2, x3], dim=1)
        return X_cf

# ==========================================
# 2. Geometric Representation Learner
# ==========================================
class GeomFairAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        # ELU allows for non-zero second derivatives (unlike ReLU which is 0 or inf)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ELU(),
            nn.Linear(32, 64), nn.ELU(),
            nn.Linear(64, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ELU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        y_pred = self.classifier(z)
        return x_rec, y_pred, z

    def decode(self, z):
        return self.decoder(z)

# ==========================================
# 3. Geometric Computation Engine (FIXED)
# ==========================================
def compute_geometric_loss(model, z, z_cf, calc_hessian=False):
    """
    Computes Metric Tensor (1st order) and optional Hessian (2nd order) differences.
    FIX: Handles vector-valued Hessian computation by iterating output dims.
    """
    batch_size = z.shape[0]
    
    # Optimization: Only compute geometry on a subset to speed up training
    subset_size = 8 if calc_hessian else 16
    subset_idx = torch.randperm(batch_size)[:subset_size]
    z_sub = z[subset_idx]
    z_cf_sub = z_cf[subset_idx]
    
    metric_loss = 0.0
    curvature_loss = 0.0
    
    # Pre-determine output dimension (usually 3 for this dataset)
    dummy_out = model.decode(z_sub[0:1])
    output_dim = dummy_out.shape[1]
    
    for i in range(len(z_sub)):
        z_i = z_sub[i].unsqueeze(0)
        z_cf_i = z_cf_sub[i].unsqueeze(0)
        
        # --- 1. Jacobian & Metric Tensor ---
        J = F.jacobian(model.decode, z_i, create_graph=True).squeeze()
        J_cf = F.jacobian(model.decode, z_cf_i, create_graph=True).squeeze()
        
        G = torch.matmul(J.T, J)
        G_cf = torch.matmul(J_cf.T, J_cf)
        metric_loss += torch.norm(G - G_cf, p='fro')
        
        # --- 2. Hessian (Curvature) ---
        if calc_hessian:
            # We must compute Hessian for EACH output dimension k separately
            # because F.hessian expects a scalar output.
            H_list = []
            H_cf_list = []
            
            for k in range(output_dim):
                # Define scalar wrapper for the k-th output dimension
                def scalar_decode(z_in):
                    return model.decode(z_in).squeeze()[k]

                h_k = F.hessian(scalar_decode, z_i, create_graph=True).squeeze()
                h_cf_k = F.hessian(scalar_decode, z_cf_i, create_graph=True).squeeze()
                
                H_list.append(h_k)
                H_cf_list.append(h_cf_k)
            
            # Stack to get full 3rd order tensor: (Out_Dim, Latent, Latent)
            H = torch.stack(H_list)
            H_cf = torch.stack(H_cf_list)
            
            curvature_loss += torch.norm(H - H_cf, p='fro')

    return metric_loss / len(z_sub), curvature_loss / len(z_sub)

# ==========================================
# 4. Evaluation Function (The Metrics)
# ==========================================
def evaluate_tradeoff(scm, model, model_name):
    """
    Calculates separate Utility and Fairness metrics.
    """
    print(f"\nEvaluating {model_name}...")
    X, A, Y, U = scm.sample(200) # Test set
    
    # Generate Counterfactuals
    X_cf_0 = scm.get_counterfactual(X, A, U, 0)
    X_cf_1 = scm.get_counterfactual(X, A, U, 1)
    X_cf = torch.where(A.bool(), X_cf_0, X_cf_1) # Flip A
    
    # 1. Forward Pass (Utility)
    with torch.no_grad():
        rec, pred, z = model(X)
        _, _, z_cf = model(X_cf)
        
        # Utility Metric 1: Reconstruction MSE
        mse = nn.MSELoss()(rec, X).item()
        
        # Utility Metric 2: Classification Accuracy
        acc = ((pred > 0.5).float() == Y).float().mean().item()
        
        # Fairness Metric 1: Latent Shift (Pointwise)
        latent_shift = torch.mean(torch.norm(z - z_cf, dim=1)).item()

    # 2. Geometric Pass (Fairness) - Requires Gradients
    # We calculate this on a small subset to be fast but accurate
    metric_err, curv_err = compute_geometric_loss(model, z[:16], z_cf[:16], calc_hessian=True)
    
    results = {
        "Model": model_name,
        "Utility_MSE_Rec": mse,
        "Utility_Accuracy": acc,
        "Fairness_Latent_Shift": latent_shift,
        "Fairness_Metric_Error": metric_err.item(),
        "Fairness_Curv_Error": curv_err.item()
    }
    
    return results

# ==========================================
# 5. Training Loop
# ==========================================
def train_model(scm, model_type="CMF", epochs=300, lr=1e-3, 
                lambda_metric=5.0, lambda_curv=1.0):
    
    model = GeomFairAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X, A, Y, U = scm.sample(1000)
    
    # Prepare counterfactuals
    X_cf_0 = scm.get_counterfactual(X, A, U, 0)
    X_cf_1 = scm.get_counterfactual(X, A, U, 1)
    X_counter = torch.where(A.bool(), X_cf_0, X_cf_1)

    dataset = TensorDataset(X, Y, X_counter)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Training {model_type}...")
    
    for epoch in range(epochs):
        for batch_X, batch_Y, batch_X_cf in loader:
            optimizer.zero_grad()
            
            rec_X, pred_Y, z = model(batch_X)
            _, _, z_cf = model(batch_X_cf)
            
            # Standard Losses
            loss_rec = nn.MSELoss()(rec_X, batch_X)
            loss_clf = nn.BCELoss()(pred_Y, batch_Y)
            loss_align = nn.MSELoss()(z, z_cf) # Anchor point alignment
            
            loss = loss_rec + loss_clf + loss_align
            
            # CMF Losses
            if model_type == "CMF":
                # Calculate Hessian during training is expensive but powerful
                l_metric, l_curv = compute_geometric_loss(model, z, z_cf, calc_hessian=True)
                loss += (lambda_metric * l_metric) + (lambda_curv * l_curv)
            
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} complete")
            
    return model

# ==========================================
# 6. Visualization
# ==========================================
def visualize_results(scm, model, title, filename):
    X, A, Y, U = scm.sample(500)
    with torch.no_grad():
        _, _, z = model(X)
    
    z = z.numpy()
    A = A.numpy().flatten()
    U = U.numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Sensitive Attribute Separation
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=A, palette="coolwarm", ax=axes[0], alpha=0.8)
    axes[0].set_title(f"{title}\nColored by Sensitive Attr (A) - Should be Mixed")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Intrinsic Manifold Structure
    scatter = axes[1].scatter(z[:,0], z[:,1], c=U, cmap="viridis", alpha=0.8)
    axes[1].set_title(f"{title}\nColored by Intrinsic (U) - Should be Smooth")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label="Intrinsic Manifold Coord (U)")
    
    plt.tight_layout()
    plt.savefig(f"results/{filename}.png", dpi=300)
    plt.close()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    scm = CausalManifoldSCM()
    
    # 1. Train Baseline
    print("--- 1. Baseline Model ---")
    base_model = train_model(scm, model_type="Baseline", epochs=200)
    visualize_results(scm, base_model, "Baseline (Vanilla AE)", "fig_baseline")
    res_base = evaluate_tradeoff(scm, base_model, "Baseline")
    
    # 2. Train CMF
    print("\n--- 2. Causal Manifold Fairness (CMF) ---")
    cmf_model = train_model(scm, model_type="CMF", epochs=200, 
                            lambda_metric=10.0, lambda_curv=5.0)
    visualize_results(scm, cmf_model, "Causal Manifold Fairness", "fig_cmf")
    res_cmf = evaluate_tradeoff(scm, cmf_model, "CMF")
    
    # 3. Print Comparison Table
    df = pd.DataFrame([res_base, res_cmf])
    print("\n" + "="*50)
    print("FINAL RESULTS: FAIRNESS-UTILITY TRADE-OFF")
    print("="*50)
    print(df.to_string(index=False))
    
    # Save metrics to file
    df.to_csv("results/metrics.csv", index=False)
    print("\nMetrics saved to results/metrics.csv")
    print("Images saved to results/ folder.")
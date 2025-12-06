import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. The Causal Environment (SCM)
# ==========================================
class CausalManifoldSCM:
    """
    Simulates data lying on a manifold warped by a sensitive attribute A.
    
    Structure:
    A (Sensitive) ~ Bernoulli(0.5)
    U (Intrinsic Latent) ~ Uniform(0, 4*pi)
    
    X is a 'Warped Swiss Roll':
    The frequency and spread of the roll depend on A.
    """
    def __init__(self):
        pass

    def sample(self, n_samples=1000):
        # Sensitive Attribute
        A = torch.randint(0, 2, (n_samples, 1)).float()
        
        # Latent intrinsic structure (the true manifold coordinate)
        U = torch.rand(n_samples, 1) * 4 * np.pi
        
        # Generate Observables X based on Structural Equations
        # If A=1, the manifold is tighter and twisted differently
        
        # Warp factor based on A
        w = 1.0 + 0.5 * A 
        
        x1 = (U * w) * torch.cos(U)
        x2 = (U * w) * torch.sin(U)
        x3 = U + (A * 2.0) # A shifts the data in 3rd dimension too
        
        X = torch.cat([x1, x2, x3], dim=1)
        
        # Target Y: Depends on intrinsic U, but biased by A
        # We want to predict Y, but Y is casually dependent on U (fair) and A (unfair)
        # For this toy example, let's say Y should depend on U only (fair label)
        # but in raw data, it correlates with A.
        Y = (torch.sin(U) > 0).float()
        
        return X, A, Y, U

    def get_counterfactual(self, X, A, U, target_a_val):
        """
        Intervention: do(A = target_a_val)
        We keep U fixed (the essence of the individual) and regenerate X.
        """
        n = X.shape[0]
        A_cf = torch.ones_like(A) * target_a_val
        
        # Re-run structural equations with A_cf and original U
        w = 1.0 + 0.5 * A_cf
        
        x1 = (U * w) * torch.cos(U)
        x2 = (U * w) * torch.sin(U)
        x3 = U + (A_cf * 2.0)
        
        X_cf = torch.cat([x1, x2, x3], dim=1)
        return X_cf, A_cf

# ==========================================
# 2. Geometric Representation Learner
# ==========================================
class GeomFairAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: X -> z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: z -> X_rec
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, input_dim)
        )
        
        # Predictor: z -> Y_hat
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        y_pred = self.classifier(z)
        return x_rec, y_pred, z

    def decode(self, z):
        return self.decoder(z)

# ==========================================
# 3. Riemannian Metric Computation
# ==========================================
def compute_metric_tensor(model, z_batch):
    """
    Computes the Pullback Metric Tensor G = J^T * J
    Where J is the Jacobian of the Decoder w.r.t z.
    
    This measures how distances in Z-space map to distances in X-space.
    """
    batch_size, latent_dim = z_batch.shape
    
    # We need to compute gradients per sample. 
    # This loop is inefficient for massive batches but clear for logic.
    # For production, use torch.vmap (functorch).
    
    metrics = []
    
    for i in range(batch_size):
        # Get single latent vector
        z_i = z_batch[i].unsqueeze(0) 
        
        # Compute Jacobian of Decoder(z) w.r.t z
        # func: z -> x
        # J shape: [1, input_dim, 1, latent_dim]
        J = F.jacobian(model.decode, z_i, create_graph=True)
        J = J.squeeze() # [input_dim, latent_dim]
        
        # Riemannian Metric: G = J^T @ J
        G = torch.matmul(J.T, J) # [latent_dim, latent_dim]
        metrics.append(G)
        
    return torch.stack(metrics) # [Batch, Latent, Latent]

# ==========================================
# 4. Training Logic with CMF Loss
# ==========================================
def train_model(scm, model_type="CMF", epochs=500, lr=1e-3, geom_lambda=5.0):
    model = GeomFairAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generate Training Data
    X, A, Y, U = scm.sample(1000)
    
    # Prepare Counterfactuals for Training (Simulating Known SCM)
    # We create the counterfactual input for every sample by flipping A
    A_flipped = 1 - A
    X_cf, _ = scm.get_counterfactual(X, A, U, 1) # This is simplified, strictly we flip
    # Let's do exact pairing:
    X_cf_0, _ = scm.get_counterfactual(X, A, U, 0)
    X_cf_1, _ = scm.get_counterfactual(X, A, U, 1)
    
    # If A=0, X_cf is X_cf_1. If A=1, X_cf is X_cf_0.
    X_counter = torch.where(A.bool(), X_cf_0, X_cf_1)

    dataset = TensorDataset(X, Y, X_counter)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    history = []

    print(f"Training {model_type}...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_X, batch_Y, batch_X_cf in loader:
            optimizer.zero_grad()
            
            # --- Forward Pass (Factual) ---
            rec_X, pred_Y, z = model(batch_X)
            
            # --- Forward Pass (Counterfactual) ---
            # We only need z_cf for the fairness loss
            z_cf = model.encoder(batch_X_cf)
            
            # 1. Reconstruction Loss
            loss_rec = nn.MSELoss()(rec_X, batch_X)
            
            # 2. Classification Loss
            loss_clf = nn.BCELoss()(pred_Y, batch_Y)
            
            loss = loss_rec + loss_clf
            
            # 3. Causal Manifold Fairness Loss
            if model_type == "CMF":
                # Compute Metric Tensors
                G_factual = compute_metric_tensor(model, z)
                G_counter = compute_metric_tensor(model, z_cf)
                
                # Geometric Invariance: 
                # The local curvature/scaling shouldn't change just because A changed.
                # Use Frobenius norm of the difference between metric tensors
                loss_geom = torch.mean(torch.norm(G_factual - G_counter, p='fro', dim=(1,2)))
                
                # Optional: Representation alignment (standard CF)
                # Helps convergence to anchor the points nearby
                loss_align = nn.MSELoss()(z, z_cf)
                
                loss += (geom_lambda * loss_geom) + (loss_align)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f}")
            
    return model

# ==========================================
# 5. Visualization & Evaluation
# ==========================================
def visualize_results(scm, model, title):
    # Generate test data
    X, A, Y, U = scm.sample(500)
    
    with torch.no_grad():
        _, _, z = model(X)
        
    z = z.numpy()
    A = A.numpy().flatten()
    U = U.numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Latent Space colored by Sensitive Attribute A
    # If fair, distributions should overlap/align perfectly
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=A, palette="coolwarm", ax=axes[0], alpha=0.7)
    axes[0].set_title(f"{title}: Latent Space by Sensitive Attr (A)")
    
    # Plot 2: Latent Space colored by Intrinsic U
    # We want to preserve U structure (the manifold shape)
    scatter = axes[1].scatter(z[:,0], z[:,1], c=U, cmap="viridis", alpha=0.7)
    axes[1].set_title(f"{title}: Latent Space by Intrinsic (U)")
    plt.colorbar(scatter, ax=axes[1], label="Intrinsic U")
    
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    scm = CausalManifoldSCM()
    
    # 1. Train Baseline (Standard Autoencoder + Classifier)
    # The latent space will likely separate based on A because A changes X heavily.
    baseline_model = train_model(scm, model_type="Baseline", epochs=400)
    visualize_results(scm, baseline_model, "Baseline")
    
    # 2. Train CMF Model (Enforces Riemannian Metric Invariance)
    # The latent space should align A=0 and A=1 while preserving the curve of U.
    cmf_model = train_model(scm, model_type="CMF", epochs=400, geom_lambda=10.0)
    visualize_results(scm, cmf_model, "Causal Manifold Fairness")

    # 3. Quantitative Check (Metric Stability)
    # Check how much the metric tensor changes upon intervention
    X_test, A_test, _, U_test = scm.sample(10)
    X_cf_test, _ = scm.get_counterfactual(X_test, A_test, U_test, 1 - A_test[0].item()) # flip
    
    z_base = baseline_model.encoder(X_test)
    z_base_cf = baseline_model.encoder(X_cf_test)
    diff_base = torch.norm(compute_metric_tensor(baseline_model, z_base) - 
                           compute_metric_tensor(baseline_model, z_base_cf), p='fro', dim=(1,2)).mean()
    
    z_cmf = cmf_model.encoder(X_test)
    z_cmf_cf = cmf_model.encoder(X_cf_test)
    diff_cmf = torch.norm(compute_metric_tensor(cmf_model, z_cmf) - 
                          compute_metric_tensor(cmf_model, z_cmf_cf), p='fro', dim=(1,2)).mean()
    
    print("="*30)
    print(f"Geometric Invariance Error (Lower is better):")
    print(f"Baseline: {diff_base.item():.4f}")
    print(f"CMF (Ours): {diff_cmf.item():.4f}")
    print("="*30)
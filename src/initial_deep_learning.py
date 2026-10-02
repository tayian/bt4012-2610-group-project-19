# ==============================================================================
# STEG Fraud Analytics Pipeline
# 1. Preprocessing & Client-Level Aggregation
# 2. Feature Engineering (Index Deltas, Abnormal Statues, Trends)
# 3. Class-Weighted Logistic Regression Baseline
# 4. Deep Learning Autoencoder (PyTorch) for Anomaly Detection
# 5. Budget-Based Inspection Evaluation (Top 1%, 5%, 10%)
# ==============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------------
# STEP 1: LOAD & AGGREGATE DATA
# ------------------------------------------------------------------------------
print("Loading datasets...")
client_df = pd.read_csv('Data/client_train.csv')
invoice_df = pd.read_csv('Data/invoice_train.csv')

print(f"Clients shape: {client_df.shape}, Invoices shape: {invoice_df.shape}")

# Feature Engineering on Invoices
invoice_df['index_delta'] = invoice_df['new_index'] - invoice_df['old_index']
invoice_df['total_declared_cons'] = (
    invoice_df['consommation_level_1'] + 
    invoice_df['consommation_level_2'] + 
    invoice_df['consommation_level_3'] + 
    invoice_df['consommation_level_4']
)

# Indicator for impossible readings or index-reconciliation mismatch
invoice_df['is_negative_delta'] = (invoice_df['index_delta'] < 0).astype(int)
invoice_df['is_reconciliation_mismatch'] = (invoice_df['index_delta'] != invoice_df['total_declared_cons']).astype(int)
invoice_df['is_zero_consumption'] = (invoice_df['total_declared_cons'] == 0).astype(int)

# Group Invoices to Client Level
print("Aggregating invoice histories to client level...")
inv_agg = invoice_df.groupby('client_id').agg(
    invoice_count=('invoice_date', 'count'),
    cons_l1_mean=('consommation_level_1', 'mean'),
    cons_l1_max=('consommation_level_1', 'max'),
    cons_l1_std=('consommation_level_1', 'std'),
    total_cons_mean=('total_declared_cons', 'mean'),
    total_cons_max=('total_declared_cons', 'max'),
    total_cons_std=('total_declared_cons', 'std'),
    neg_delta_count=('is_negative_delta', 'sum'),
    mismatch_count=('is_reconciliation_mismatch', 'sum'),
    zero_cons_count=('is_zero_consumption', 'sum'),
    months_number_mean=('months_number', 'mean')
).reset_index()

# Fill NaN std values (for clients with 1 invoice)
inv_agg = inv_agg.fillna(0)

# Merge Aggregated Invoice Features with Client Table
data = pd.merge(client_df, inv_agg, on='client_id', how='left')

# Encode Client Features
data['client_catg'] = data['client_catg'].astype('category').cat.codes
data['disrict'] = data['disrict'].astype('category').cat.codes
data['region'] = data['region'].astype('category').cat.codes

feature_cols = [c for c in data.columns if c not in ['client_id', 'target', 'creation_date']]
X = data[feature_cols].values
y = data['target'].values

# Stratified Client-Level Train/Holdout Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# STEP 2: CLASS-WEIGHTED LOGISTIC REGRESSION BASELINE
# ------------------------------------------------------------------------------
print("\n--- Training Class-Weighted Logistic Regression ---")
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_lr)
pr_auc_lr = auc(recall, precision)
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"Logistic Regression PR-AUC:  {pr_auc_lr:.4f}")
print(f"Logistic Regression ROC-AUC: {roc_auc_lr:.4f}")

# ------------------------------------------------------------------------------
# STEP 3: DEEP LEARNING AUTOENCODER FOR ANOMALY DETECTION (PyTorch)
# ------------------------------------------------------------------------------
print("\n--- Training PyTorch Autoencoder on Normal (Non-Fraud) Clients ---")

# Train Autoencoder ONLY on normal data (y = 0)
X_train_normal = X_train_scaled[y_train == 0]
train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(FraudAutoencoder, self).__init__()
        # Encoder: compress features to low-dimensional bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder: reconstruct original features
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = X_train_scaled.shape[1]
autoencoder = FraudAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
epochs = 15
autoencoder.train()
for epoch in range(epochs):
    loss_sum = 0.0
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_sum/len(X_train_normal):.6f}")

# Compute Reconstruction Error (MSE) as Fraud Risk Score
autoencoder.eval()
with torch.no_grad():
    reconstructions = autoencoder(test_tensor)
    mse_error = torch.mean((test_tensor - reconstructions) ** 2, dim=1).numpy()

pr_ae, rec_ae, _ = precision_recall_curve(y_test, mse_error)
pr_auc_ae = auc(rec_ae, pr_ae)
roc_auc_ae = roc_auc_score(y_test, mse_error)

print(f"Autoencoder PR-AUC:  {pr_auc_ae:.4f}")
print(f"Autoencoder ROC-AUC: {roc_auc_ae:.4f}")

# ------------------------------------------------------------------------------
# STEP 4: BUDGET-BASED EVALUATION (PRECISION & RECALL @ TOP K%)
# ------------------------------------------------------------------------------
def evaluate_budget(y_true, scores, k_percentages=[0.01, 0.05, 0.10]):
    n_clients = len(y_true)
    sorted_indices = np.argsort(scores)[::-1]  # Rank highest risk to lowest
    
    print("\nOperational Performance under Fixed Inspection Budgets:")
    for k in k_percentages:
        top_n = int(n_clients * k)
        top_indices = sorted_indices[:top_n]
        
        tp = np.sum(y_true[top_indices] == 1)
        fp = np.sum(y_true[top_indices] == 0)
        fn = np.sum(y_true) - tp
        
        precision_at_k = tp / top_n
        recall_at_k = tp / np.sum(y_true)
        
        print(f"  Top {int(k*100)}% Budget ({top_n} clients inspected):")
        print(f"    - Precision @ {int(k*100)}%: {precision_at_k:.4f}")
        print(f"    - Recall @ {int(k*100)}%:    {recall_at_k:.4f}")
        print(f"    - Fraud Cases Caught: {tp} / {np.sum(y_true)}")

print("\n--- Logistic Regression Inspection Ranking ---")
evaluate_budget(y_test, y_pred_proba_lr)

print("\n--- Autoencoder Anomaly Score Inspection Ranking ---")
evaluate_budget(y_test, mse_error)
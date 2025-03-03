import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import pandas as pd

# Load DP-protected data with corrected path
df_dp = pd.read_csv(r"D:\Winter Term 2025\ELG5295-W00(Ethics,AI & Robotics)\DP_Logistic_Regression_Project\data\dp_protected_cancer_data.csv")

# Convert data to PyTorch tensors
X = torch.tensor(df_dp.drop(columns=["Diagnosis"]).values, dtype=torch.float32)  # Change "Target" to "Diagnosis"
y = torch.tensor(df_dp["Diagnosis"].values, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define Logistic Regression Model
class LogisticRegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionNN, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Initialize model
model = LogisticRegressionNN(X.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Apply Opacus for DP-SGD training with fixed parameters
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=1.0,  # Adjust privacy budget
    target_delta=1e-5,
    max_grad_norm=1.0,  # 🔥 Fix: Added required parameter
    epochs=10
)

# Train with DP-SGD
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

print("✅ DP-SGD training complete.")

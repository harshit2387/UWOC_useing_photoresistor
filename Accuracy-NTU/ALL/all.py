import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import filedialog, messagebox

# -------- 1. GUI File Picker --------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
if not file_path:
    messagebox.showerror("Error", "No file selected. Exiting.")
    exit()

# -------- 2. Load and Reshape Wide Data --------
df_wide = pd.read_csv(file_path, skiprows=1)  # Skip NTU header row
df_wide = df_wide.rename(columns={"lables.3": "labels.3"})  # Fix typo

# Extract NTU column pairs
ntu_columns = []
columns = df_wide.columns.tolist()
for i in range(0, len(columns), 2):
    label_col = columns[i]
    feature_col = columns[i + 1]
    ntu_columns.append((label_col, feature_col))

# Generate NTU range labels like "1-2 NTU", "2-3 NTU", ...
ntu_ranges = [f"{i+1},{i+2} NTU" for i in range(len(ntu_columns))]

# Combine all NTU segments into one long DataFrame
ntu_data = []
for (label_col, feature_col), ntu in zip(ntu_columns, ntu_ranges):
    temp_df = df_wide[[label_col, feature_col]].copy()
    temp_df.columns = ["labels", "feature"]
    temp_df["NTU Category"] = ntu
    ntu_data.append(temp_df)

df_long = pd.concat(ntu_data, ignore_index=True)
df_long = df_long.dropna()
df_long['labels'] = df_long['labels'].astype(int)
df_long['feature'] = df_long['feature'].astype(float)

# -------- 3. Normalize Feature --------
scaler = MinMaxScaler()
X = scaler.fit_transform(df_long[['feature']].values.astype(np.float32))
y = df_long['labels'].values.astype(np.int64)

# -------- 4. Train-Test Split --------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- 5. PyTorch Dataset --------
class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CSVDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# -------- 6. Define Model --------
class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleNet(input_dim=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------- 7. Train Model --------
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# -------- 8. Predict on Full Dataset --------
model.eval()
with torch.no_grad():
    inputs = torch.tensor(X).float()
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).numpy()

df_long['predictions'] = preds

# -------- 9. Compute Recall per NTU Category --------
recall_by_ntu = (
    df_long.groupby('NTU Category')
    .apply(lambda g: recall_score(g['labels'], g['predictions'], zero_division=0))
    .reset_index(name='Recall')
)

# -------- 10. Plot Bar Chart --------
plt.figure(figsize=(10, 6))
plt.bar(recall_by_ntu['NTU Category'], recall_by_ntu['Recall'], color='salmon', edgecolor='black')
plt.xlabel("NTU Category")
plt.ylabel("Recall")
plt.title("Recall vs NTU Category")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
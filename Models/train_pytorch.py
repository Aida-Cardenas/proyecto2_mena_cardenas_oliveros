# src/models/train_pytorch.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GamesDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).float().unsqueeze(1) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PerceptronBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, 1) 
        )

    def forward(self, x):
        return self.net(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    logits_arr = np.vstack(all_logits)
    labels_arr = np.vstack(all_labels)
    probs = 1 / (1 + np.exp(-logits_arr))  
    preds = (probs >= 0.5).astype(int)
    return total_loss / len(dataloader.dataset), labels_arr, preds, probs

def main():
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 50
    DROPOUT = 0.2
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    X_df = pd.read_csv("Data/Processed/features_matrix.csv")
    y_df = pd.read_csv("Data/Processed/labels.csv")

    y = y_df.values.squeeze()

    X = X_df.values.astype(float)
    if np.isnan(X).any():
        print("Se encontraron NaN en features. Imputando con medianas por columna.")
        col_medias = np.nanmedian(X, axis=0)
        inds_nan = np.where(np.isnan(X))
        X[inds_nan] = np.take(col_medias, inds_nan[1])

    if np.isinf(X).any():
        print("Se encontraron valores infinitos en features. Reemplazando con 0.")
        X[np.isinf(X)] = 0.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    train_ds = GamesDataset(X_train, y_train)
    val_ds = GamesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]

    print("\n=== Entrenando Perceptron Baseline ===")
    baseline = PerceptronBaseline(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(baseline.parameters(), lr=LR)

    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(baseline, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred, y_prob = evaluate(baseline, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            os.makedirs("Models", exist_ok=True)
            torch.save(baseline.state_dict(), "Models/baseline_perceptron.pth")
        else:
            no_improve += 1
            if no_improve >= 5:
                print("No mejora en validación, deteniendo entrenamiento early.")
                break

    val_loss, y_true, y_pred, y_prob = evaluate(baseline, val_loader, criterion, device)
    print("\n=== Resultados Baseline ===")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    try:
        if not np.isnan(y_prob).any() and len(np.unique(y_true)) == 2:
            print("ROC-AUC:", round(roc_auc_score(y_true, y_prob), 3))
        else:
            print("ROC-AUC: No aplicable (NaN o clase única)")
    except Exception:
        print("ROC-AUC: No aplicable (error al calcular)")

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    print("\n=== Entrenando MLPClassifier ===")
    mlp = MLPClassifier(input_dim, hidden1=64, hidden2=32, dropout_rate=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(mlp, train_loader, criterion, optimizer, device)
        val_loss, y_true_mlp, y_pred_mlp, y_prob_mlp = evaluate(mlp, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(mlp.state_dict(), "Models/mlp_classifier.pth")
        else:
            no_improve += 1
            if no_improve >= 7:
                print("No mejora en validación, deteniendo entrenamiento early.")
                break

    val_loss, y_true_mlp, y_pred_mlp, y_prob_mlp = evaluate(mlp, val_loader, criterion, device)
    print("\n=== Resultados MLPClassifier ===")
    print("Classification Report:")
    print(classification_report(y_true_mlp, y_pred_mlp))

    try:
        if not np.isnan(y_prob_mlp).any() and len(np.unique(y_true_mlp)) == 2:
            print("ROC-AUC:", round(roc_auc_score(y_true_mlp, y_prob_mlp), 3))
        else:
            print("ROC-AUC: No aplicable (NaN o clase única)")
    except Exception:
        print("ROC-AUC: No aplicable (error al calcular)")

    print("Confusion Matrix:\n", confusion_matrix(y_true_mlp, y_pred_mlp))

    print("\nPesos finales guardados en:")
    print("- Models/baseline_perceptron.pth")
    print("- Models/mlp_classifier.pth")

if __name__ == "__main__":
    main()

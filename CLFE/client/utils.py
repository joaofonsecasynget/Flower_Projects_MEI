# NOTA: O particionamento dos dados é agora controlado por dois parâmetros distintos:
#   - num_clients: número de clientes a executar
#   - num_total_clients: número total de clientes para dividir o dataset (ver client.py)
# O código principal do particionamento está em client.py.

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearClassificationModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        if isinstance(features, np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = torch.tensor(features.values, dtype=torch.float32)
        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = torch.tensor(targets.values, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train(model, dataloader, epochs=1, device="cpu"):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"[TRAIN] Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, device="cpu"):
    criterion = nn.BCELoss()
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets.view(-1, 1))
            losses.append(loss.item())
            preds.extend(outputs.cpu().numpy().flatten())
            trues.extend(targets.cpu().numpy().flatten())
    
    # Cálculo de métricas de classificação
    avg_loss = np.mean(losses)
    pred_classes = np.array(preds) >= 0.5
    true_classes = np.array(trues) >= 0.5
    
    # Cálculo de métricas de classificação
    accuracy = np.mean(pred_classes == true_classes)
    
    # Precision, recall, F1 (evitando divisão por zero)
    true_positives = np.sum((pred_classes == 1) & (true_classes == 1))
    predicted_positives = np.sum(pred_classes == 1)
    actual_positives = np.sum(true_classes == 1)
    
    precision = true_positives / max(predicted_positives, 1)
    recall = true_positives / max(actual_positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Ainda calculamos RMSE para compatibilidade
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2))
    
    return avg_loss, rmse, preds, trues, accuracy, precision, recall, f1

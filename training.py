import logging
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model': 'CBF',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'seed': 42,
    'data_path': 'data.csv',
    'log_path': 'logs',
    'model_path': 'models'
}

# Exception classes
class TrainingError(Exception):
    pass

class DataError(Exception):
    pass

# Data structures/models
class CBFModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(CBFModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh(self.fc3(x))
        return x

class Data(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Utility methods
def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = pd.read_csv(data_path)
        X = data.drop('label', axis=1).values
        y = data['label'].values
        return X, y
    except Exception as e:
        raise DataError(f'Failed to load data: {e}')

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.fit_transform(X)

def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = Data(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model: CBFModel, device: torch.device, X_train: np.ndarray, y_train: np.ndarray, batch_size: int, epochs: int, learning_rate: float) -> None:
    scaler = StandardScaler()
    X_train_scaled = scale_data(X_train, scaler)
    data_loader = create_data_loader(X_train_scaled, y_train, batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')
    torch.save(model.state_dict(), os.path.join(CONFIG['model_path'], 'model.pth'))

def evaluate_model(model: CBFModel, device: torch.device, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float]:
    scaler = StandardScaler()
    X_test_scaled = scale_data(X_test, scaler)
    X_test_scaled = torch.from_numpy(X_test_scaled).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_scaled)
        loss = nn.MSELoss()(outputs, y_test.view(-1, 1))
        accuracy = accuracy_score(y_test.cpu().numpy(), torch.round(outputs).cpu().numpy())
        f1 = f1_score(y_test.cpu().numpy(), torch.round(outputs).cpu().numpy())
        precision = precision_score(y_test.cpu().numpy(), torch.round(outputs).cpu().numpy())
        recall = recall_score(y_test.cpu().numpy(), torch.round(outputs).cpu().numpy())
    return loss.item(), accuracy, f1, precision, recall

def main() -> None:
    # Load data
    X, y = load_data(CONFIG['data_path'])
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create model
    model = CBFModel(X_train.shape[1], CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Train model
    train_model(model, device, X_train, y_train, CONFIG['batch_size'], CONFIG['epochs'], CONFIG['learning_rate'])
    
    # Evaluate model
    loss, accuracy, f1, precision, recall = evaluate_model(model, device, X_test, y_test)
    logger.info(f'Test Loss: {loss}, Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.flattened_size = 64 * 2 * 2  
        
        self.embedding_dim = self.flattened_size  
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.fc2 = nn.Linear(128, 1)  
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  
        
        x = x.view(x.size(0), -1)  
        x = x.unsqueeze(1)  
        
        x = self.transformer_encoder(x)
        
        x = F.relu(self.fc1(x[:, 0, :]))  # Use the output of the first position
        x = self.fc2(x)
        
        return x



def train_model_transformer(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            predictions = torch.round(torch.sigmoid(outputs))
            total_train += y_batch.size(0)
            correct_train += (predictions == y_batch).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch).squeeze(1)
                loss = criterion(outputs, y_val_batch)
                
                val_loss += loss.item() * X_val_batch.size(0)
                predictions = torch.round(torch.sigmoid(outputs))
                total_val += y_val_batch.size(0)
                correct_val += (predictions == y_val_batch).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

def data_prep_transformer(feature, target, batch_size):
    feat = torch.tensor(feature, dtype=torch.float32)
    label = torch.tensor(target, dtype=torch.float32)
    data = TensorDataset(feat, label)
    loader = DataLoader(data, batch_size=batch_size)
    return loader
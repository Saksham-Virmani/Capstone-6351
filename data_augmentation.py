import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('cuda')

def gaf_preprocess(df):
    gaf = GramianAngularField(method='summation')
    df_processed = gaf.fit_transform(df)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_processed.reshape(-1, df_processed.shape[-1])).reshape(df_processed.shape)
    
    return df_scaled

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2_mean = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x, labels):
        x = torch.cat([x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 1, 256)
        self.fc2 = nn.Linear(256, 128 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=1)
    
    def forward(self, x, labels):
        labels = labels.view(-1, 1)
        x = torch.cat([x, labels], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 3, 3)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, labels):
        mean, logvar = self.encoder(x, labels)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z, labels)
        return recon_x, mean, logvar

def loss_function(recon_x, x, mean, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD


def get_tensor(ndarray, labels):
    tensor = torch.tensor(ndarray, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return tensor, labels


def train_model_cvae(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.6f}')

def create_model(latent_dim, device):
    model = CVAE(latent_dim).to(device)
    return model

def create_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def synthetic_data(model, labels, n=2560):
    model.eval()
    with torch.no_grad():
        # Generate the latent vectors with the correct batch size
        z = torch.randn(n, model.encoder.fc2_mean.out_features).to(device)
        print(f"z shape: {z.shape}")
        
        # Convert the labels to a tensor and ensure they have the correct shape
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        print(f"labels shape: {labels.shape}")
        
        # Generate synthetic data using the decoder
        synthetic_data = model.decoder(z, labels)
        print(f"synthetic_data shape: {synthetic_data.shape}")
        synthetic_data = synthetic_data.cpu().numpy()
        return synthetic_data


def prepare_data_with_labels(data, target):
    tensor, labels = get_tensor(data, target)
    dataset = DataLoader(TensorDataset(tensor, labels))
    return dataset
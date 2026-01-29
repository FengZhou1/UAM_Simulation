import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from st_gat_model import STGAT

def train_model(data_path, model_path='stgat_model.pth', epochs=10):
    print("Loading data...")
    if not os.path.exists(data_path):
        print("Data file not found. Run simulation in data_collection mode first.")
        return

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Data format: (num_samples, num_nodes, n_features)
    # We need to construct sequences
    raw_data = np.load(data_path, allow_pickle=True).item()
    features = raw_data['features'] # (T, N, F)
    adj = raw_data['adj']
    
    # Preprocess Adjacency Matrix
    adj = torch.FloatTensor(adj).to(device)
    
    # Create Sequences
    seq_len = 5
    prediction_horizon = 10 # Predict 10 steps ahead instead of 1
    
    X = []
    Y = []
    
    num_timesteps = features.shape[0]
    num_nodes = features.shape[1]
    
    # Normalize features
    # Simple MinMax or Z-score
    feat_mean = np.mean(features, axis=(0,1))
    feat_std = np.std(features, axis=(0,1))
    features = (features - feat_mean) / (feat_std + 1e-5)
    
    for i in range(num_timesteps - seq_len - prediction_horizon):
        x_seq = features[i : i+seq_len]
        y_target = features[i+seq_len+prediction_horizon-1, :, 0] # Predict density (feature 0)
        X.append(x_seq)
        Y.append(y_target)
        
    X = np.array(X) 
    Y = np.array(Y)
    
    X_tensor = torch.FloatTensor(X).to(device) # (Samples, T, N, F)
    Y_tensor = torch.FloatTensor(Y).unsqueeze(-1).to(device) # (Samples, N, 1)
    
    # Model
    nfeat = X_tensor.shape[-1]
    nhid = 64
    nclass = 1 # Regression: density
    
    # Must pass n_nodes for centralized LSTM (Architecture Update)
    model = STGAT(nfeat, nhid, nclass, n_nodes=num_nodes, dropout=0.2, alpha=0.2, nheads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    model.train()
    
    batch_size = 32
    num_samples = X_tensor.shape[0]
    
    for epoch in range(epochs):
        total_loss = 0
        permutation = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], Y_tensor[indices]
            
            optimizer.zero_grad()
            output = model(batch_x, adj)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_samples:.6f}")
        
    # Save Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'adj': raw_data['adj']
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model('training_artifacts/data_grand_iter_3.npy')

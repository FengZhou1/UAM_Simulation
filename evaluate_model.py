import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from st_gat_model import STGAT

def validate_model(data_path, model_path='stgat_model.pth', output_plot='validation_results.png'):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    # Load Data
    raw_data = np.load(data_path, allow_pickle=True).item()
    features = raw_data['features'] # (T, N, F)
    adj = raw_data['adj']
    
    # Split into Train/Test (Last 20% for plotting)
    split_idx = int(features.shape[0] * 0.8)
    test_features = features[split_idx:]
    
    print(f"Test Set Shape: {test_features.shape}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model Checkpoint
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return
        
    try:
        if 'weights_only' in torch.load.__code__.co_varnames:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    # Extract Stats
    feat_mean = checkpoint.get('feat_mean', np.zeros(2))
    feat_std = checkpoint.get('feat_std', np.ones(2))
    saved_adj = checkpoint.get('adj', None)
    
    # Use saved adj for model structure if available
    if saved_adj is not None:
        adj_tensor = torch.FloatTensor(saved_adj).to(device)
        num_nodes = saved_adj.shape[0]
    else:
        adj_tensor = torch.FloatTensor(adj).to(device)
        num_nodes = adj.shape[0]
        
    print(f"Model Nodes: {num_nodes}")

    # Initialize Model
    # Params must match training
    nfeat = 2 # Density, SoC
    nhid = 64
    nclass = 1 
    
    model = STGAT(nfeat, nhid, nclass, n_nodes=num_nodes, dropout=0, alpha=0.2, nheads=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare Sequences
    seq_len = 5
    pred_horizon = 10
    
    # Normalize Test Data
    # IMPORTANT: Use TRAINING stats (from checkpoint)
    norm_features = (test_features - feat_mean) / (feat_std + 1e-5)
    
    X_list = []
    Y_gt_list = [] # Ground Truth
    Y_base_list = [] # Baseline: t-1 (Persistence)
    
    for i in range(len(norm_features) - seq_len - pred_horizon):
        x_seq = norm_features[i : i+seq_len] # Input: t-4 ... t
        
        # Target: t+horizon (Density only)
        # Note: Denormalize target for plotting
        y_target_norm = norm_features[i+seq_len+pred_horizon-1, :, 0] 
        y_target_raw = test_features[i+seq_len+pred_horizon-1, :, 0]
        
        # Baseline: value at t (Last seen) - Persistence Forecast
        # We compare "what I see now at t" vs "what happens at t+horizon"
        y_last_raw = test_features[i+seq_len-1, :, 0]
        
        X_list.append(x_seq)
        Y_gt_list.append(y_target_raw)
        Y_base_list.append(y_last_raw)
        
    if not X_list:
        print("Not enough data for sequence.")
        return

    X_tensor = torch.FloatTensor(np.array(X_list)).to(device)
    
    # Inference
    print("Running Inference...")
    predictions = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i+batch_size]
            # Output: (Batch, N, 1)
            out = model(batch_x, adj_tensor)
            predictions.append(out.cpu().numpy())
            
    pred_array = np.concatenate(predictions, axis=0) # (Samples, N, 1)
    pred_array = pred_array.squeeze(-1) # (Samples, N)
    
    # Denormalize Predictions
    # density_mean = feat_mean[0], density_std = feat_std[0]
    pred_denorm = pred_array * (feat_std[0] + 1e-5) + feat_mean[0]
    pred_denorm = np.maximum(0, pred_denorm) # ReLU constraint
    
    Y_gt = np.array(Y_gt_list)
    Y_base = np.array(Y_base_list)
    
    # --- Visualization ---
    # Pick Top 4 nodes with highest variance in Ground Truth
    variances = np.var(Y_gt, axis=0)
    top_indices = np.argsort(variances)[-4:]
    
    plt.figure(figsize=(15, 10))
    time_steps = np.arange(len(Y_gt))
    
    for i, node_idx in enumerate(top_indices):
        plt.subplot(2, 2, i+1)
        
        # Plot curves
        plt.plot(time_steps, Y_gt[:, node_idx], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(time_steps, pred_denorm[:, node_idx], 'r--', label='ST-GAT Prediction', linewidth=2)
        plt.plot(time_steps, Y_base[:, node_idx], 'g:', label='Naive (t-1)', linewidth=1.5, alpha=0.5)
        
        plt.title(f"Node {node_idx} (High Activity)")
        plt.xlabel("Time Step")
        plt.ylabel("Aircraft Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Validation plot saved to {output_plot}")
    
    # Calculate simple metrics
    mse_model = np.mean((pred_denorm - Y_gt)**2)
    mse_naive = np.mean((Y_base - Y_gt)**2)
    
    print("\n=== Quant Metrics (Test Set) ===")
    print(f"ST-GAT MSE: {mse_model:.4f}")
    print(f"Naive  MSE: {mse_naive:.4f}")
    if mse_model < mse_naive:
        print("RESULT: Model beats naive persistence! (Learned dynamics)")
    else:
        print("RESULT: Model is worse or equal to naive. (Needs improvement)")

if __name__ == "__main__":
    # Use the latest large dataset
    validate_model('training_artifacts/data_grand_iter_3.npy')

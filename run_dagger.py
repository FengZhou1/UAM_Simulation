import os
import shutil
import numpy as np
from simulation import run_simulation
from train_stgat import train_model

def merge_datasets(file_list, output_path):
    print(f"Merging datasets: {file_list} -> {output_path}")
    all_features = []
    adj = None
    
    for f in file_list:
        if not os.path.exists(f):
            print(f"Warning: File {f} not found, skipping merge.")
            continue
            
        try:
            data = np.load(f, allow_pickle=True).item()
            features = data['features'] # (T, N, F)
            if adj is None:
                adj = data['adj']
            all_features.append(features)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not all_features:
        print("No data collected to merge.")
        return False
        
    combined_features = np.concatenate(all_features, axis=0)
    np.save(output_path, {
        'features': combined_features,
        'adj': adj
    })
    print(f"Merged dataset saved. Total frames: {combined_features.shape[0]}")
    return True

def dagger_loop(rounds=3):
    # Phase 0: Warmup
    print("=== DAgger Phase 0: Warmup (Random/Heuristic w/o Model) ===")
    # Run simulation without model (using heuristic fallback in controller)
    # Collect initial data
    d0_path = "data_iter_0.npy"
    m0_path = "model_iter_0.pth"
    
    # 0. Clean up previous
    if os.path.exists(d0_path): os.remove(d0_path)
    
    # 1. Run Simulation
    print(f"Running simulation to generate {d0_path}...")
    run_simulation(mode='guided', visualize=False, routing_policy='stgat', 
                   model_path="heuristic_v0", # No model -> Heuristic
                   save_data=True, output_data=d0_path)
    
    # 2. Train Initial Model
    if os.path.exists(d0_path):
        print("Training Model M0...")
        train_model(data_path=d0_path, model_path=m0_path, epochs=10) 
    else:
        print("Error: Phase 0 data generation failed.")
        return

    current_model = m0_path
    data_files = [d0_path]
    
    # Iterative Phase
    for k in range(1, rounds + 1):
        print(f"\n=== DAgger Phase {k}: Self-Play & Aggregation ===")
        
        # 1. Run Simulation with current model
        new_data_path = f"data_iter_{k}.npy"
        print(f"Running simulation with {current_model} to generate {new_data_path}...")
        
        run_simulation(mode='guided', visualize=False, routing_policy='stgat',
                       model_path=current_model,
                       save_data=True, output_data=new_data_path)
        
        # 2. Key Step: Aggregate Data
        data_files.append(new_data_path)
        agg_data_path = f"data_agg_{k}.npy"
        success = merge_datasets(data_files, agg_data_path)
        
        if not success:
            print("Aggregation failed. Stopping.")
            break

        # 3. Retrain on Aggregated Data
        next_model = f"model_iter_{k}.pth"
        print(f"Retraining Model {next_model} on {agg_data_path}...")
        train_model(data_path=agg_data_path, model_path=next_model, epochs=30)
        
        current_model = next_model
        
    print("\nDAgger Training Complete.")
    print(f"Final Model: {current_model}")
    
    # Update default model
    if os.path.exists(current_model):
        shutil.copy(current_model, 'stgat_model.pth')
        print("Updated default 'stgat_model.pth' with final model.")

if __name__ == "__main__":
    # You can set visualize=True in run_simulation inside the loop if you want to watch,
    # but for training speed it is set to False.
    dagger_loop(rounds=3)

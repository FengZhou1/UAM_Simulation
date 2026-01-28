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

def dagger_loop(rounds=3, episodes_per_round=3):
    # Setup artifact directory
    artifact_dir = "training_artifacts"
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
        
    print(f"=== DAgger Phase 0: Warmup (Random/Heuristic w/o Model) ===")
    
    current_data_files = []
    
    # Run multiple episodes for warmup
    for ep in range(episodes_per_round):
        output_file = os.path.join(artifact_dir, f"data_phase0_ep{ep}.npy")
        current_data_files.append(output_file)
        
        # Vary seed: Base seed + episode index
        seed = 1000 + ep 
        print(f"  -> Episode {ep+1}/{episodes_per_round} (Seed: {seed})")
        
        run_simulation(mode='guided', visualize=False, routing_policy='stgat', 
                       model_path="heuristic_v0", 
                       save_data=True, output_data=output_file,
                       seed=seed, num_aircraft=np.random.randint(30, 50)) # Randomize density slightly

    m0_path = os.path.join(artifact_dir, "model_iter_0.pth")
    agg_data_path = os.path.join(artifact_dir, "data_phase0_agg.npy")
    
    # Merge Phase 0 data
    if merge_datasets(current_data_files, agg_data_path):
        print("Training Model M0...")
        train_model(data_path=agg_data_path, model_path=m0_path, epochs=30)
    else:
        print("Error: Phase 0 data generation failed.")
        return

    current_model = m0_path
    all_data_files = [agg_data_path] # Keep track of aggregated files
    
    # Iterative Phase
    for k in range(1, rounds + 1):
        print(f"\n=== DAgger Phase {k}: Self-Play & Aggregation ===")
        
        phase_files = []
        for ep in range(episodes_per_round):
            output_file = os.path.join(artifact_dir, f"data_phase{k}_ep{ep}.npy")
            phase_files.append(output_file)
            
            # Vary seed
            seed = 1000 + (k * 10) + ep
            print(f"  -> Episode {ep+1}/{episodes_per_round} (Seed: {seed})")
            
            run_simulation(mode='guided', visualize=False, routing_policy='stgat',
                           model_path=current_model,
                           save_data=True, output_data=output_file,
                           seed=seed, num_aircraft=np.random.randint(30, 50))
        
        # Merge new data first (optional, but good for tracking)
        phase_agg_path = os.path.join(artifact_dir, f"data_phase{k}_agg.npy")
        merge_datasets(phase_files, phase_agg_path)
        
        # Add to global list
        all_data_files.append(phase_agg_path)
        
        # Create Grand Master Dataset for training
        final_dataset_path = os.path.join(artifact_dir, f"data_grand_iter_{k}.npy")
        success = merge_datasets(all_data_files, final_dataset_path)
        
        if not success:
            print("Aggregation failed. Stopping.")
            break

        # 3. Retrain on Aggregated Data
        next_model = os.path.join(artifact_dir, f"model_iter_{k}.pth")
        print(f"Retraining Model {next_model} on {final_dataset_path}...")
        train_model(data_path=final_dataset_path, model_path=next_model, epochs=30)
        
        current_model = next_model
        
    print("\nDAgger Training Complete.")
    print(f"Final Model: {current_model}")
    
    # Update default model
    if os.path.exists(current_model):
        shutil.copy(current_model, 'stgat_model.pth')
        print("Updated default 'stgat_model.pth' with final model.")

if __name__ == "__main__":
    dagger_loop(rounds=3, episodes_per_round=3)

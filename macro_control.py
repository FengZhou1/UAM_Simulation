import numpy as np
import math
import os

try:
    import torch
    from st_gat_model import STGAT
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. ST-GAT model will be disabled.")

class STGATController:
    def __init__(self, airspace, config):
        self.airspace = airspace
        self.config = config
        self.static_capacity = config.SECTOR_CAPACITY
        
        self.prediction_horizon = 5 # 预测未来 5 个时间步
        
        # Load Model
        self.model = None
        self.history_buffer = [] # Store sequence of observations
        self.max_history = 5
        self.device = 'cpu'
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cpu')
            
            # Check for trained model
            self.model_path = 'stgat_model.pth'
            if os.path.exists(self.model_path):
                try:
                    checkpoint = torch.load(self.model_path)
                    
                    # Assume adj is static and saved with model or we reconstruct
                    # For safety, let's load what was saved
                    self.saved_adj = torch.FloatTensor(checkpoint['adj'])
                    self.feat_mean = checkpoint['feat_mean']
                    self.feat_std = checkpoint['feat_std']
                    
                    # Rebuild model structure (Hardcoded params must match training)
                    nfeat = 2 # Density, Avg SOC
                    nhid = 64
                    nclass = 1
                    self.model = STGAT(nfeat, nhid, nclass, dropout=0, alpha=0.2, nheads=4)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    print("ST-GAT Model loaded successfully.")
                except Exception as e:
                    print(f"Failed to load ST-GAT model: {e}")
                    self.model = None
            else:
                print("No ST-GAT model found. Using fallback heuristic.")
        else:
             print("PyTorch not available. Using fallback heuristic.")
    
    def predict_sector_density(self, aircraft_list):
        """
        [Macro Layer 1]: 密度预测 (The Brain)
        使用 ST-GAT 预测
        """
        # 1. 统计当前状态 (Density, Avg SoC)
        # Ensure node ordering matches adjacency matrix (Sorted by ID)
        sorted_nodes = sorted(list(self.airspace.graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        
        current_counts = np.zeros(len(sorted_nodes))
        current_socs = np.zeros(len(sorted_nodes))
        
        # Helper structures
        temp_counts = {n: 0 for n in sorted_nodes}
        temp_socs = {n: [] for n in sorted_nodes}
        
        for ac in aircraft_list:
            if not ac.finished:
                # Use ac.current_region_id
                rid = ac.current_region_id
                if rid in temp_counts:
                    temp_counts[rid] += 1
                    temp_socs[rid].append(ac.battery_soc)
        
        for i, rid in enumerate(sorted_nodes):
            current_counts[i] = temp_counts[rid]
            socs = temp_socs[rid]
            current_socs[i] = sum(socs)/len(socs) if socs else 1.0
            
        # Construct feature vector: [Density, SoC]
        current_features = np.stack([current_counts, current_socs], axis=1) # (N, 2)
        
        # Update History
        self.history_buffer.append(current_features)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
            
        # 2. Predict
        pred_densities = {}
        sector_states = {}
        
        # Populate basics first (fallback or for sectors mapping)
        for i, rid in enumerate(sorted_nodes):
            sector_states[rid] = {'soc': current_socs[i]}
            # Default to current count if model fails
            pred_densities[rid] = current_counts[i] 

        # ST-GAT Inference
        if self.model is not None and len(self.history_buffer) == self.max_history and TORCH_AVAILABLE:
            with torch.no_grad():
                # Prepare input
                # (Batch=1, Seq=5, Nodes=N, Feat=2)
                x_seq = np.array(self.history_buffer) #(T, N, F)
                
                # Normalize
                x_seq = (x_seq - self.feat_mean) / (self.feat_std + 1e-5)
                
                x_tensor = torch.FloatTensor(x_seq).unsqueeze(0)
                
                # Run Model
                # Need to use the Adjacency matrix the model was trained with
                # Or the current graph adj? Ideally they are same topology.
                adj = self.saved_adj 
                
                output = self.model(x_tensor, adj) # (1, N, 1)
                pred_vals = output.squeeze().numpy()
                
                # De-normalize? Assuming target was also normalized? 
                # In training: Y = features[..., 0]. Normalized.
                # So we must denormalize density
                pred_vals = pred_vals * (self.feat_std[0] + 1e-5) + self.feat_mean[0]
                
                for i, rid in enumerate(sorted_nodes):
                    pred_densities[rid] = max(0, pred_vals[i])
        
        return pred_densities, sector_states

    def update_airspace_costs(self, aircraft_list, wind_field):
        """
        [Macro Layer 2]: 动态定价与阈值调节 (Gating Policy)
        核心公式：T_dyn = T_static * beta_wind * (1 + lambda * e^-SoC)
        """
        pred_densities, sector_states = self.predict_sector_density(aircraft_list)
        
        for s_id in self.airspace.graph.nodes():
            sector_pos = self.airspace.regions[s_id]
            
            # --- A. 获取环境参数 ---
            wind_speed = wind_field.get_wind_at(sector_pos)
            avg_soc = sector_states[s_id]['soc']
            
            # --- B. 计算动态阈值 (论文核心公式) ---
            # 1. 风速因子 (beta_wind): 风越大，容量越小
            beta_wind = max(0.5, 1.0 - 0.03 * wind_speed)
            
            # 2. 能量豁免因子 (Energy Override): 电量越低，阈值越高
            # lambda = 0.8, 当 SoC < 0.2 时，因子急剧上升
            lambda_energy = 0.8
            energy_factor = 1.0 + lambda_energy * math.exp(-10 * avg_soc)
            
            # 3. 最终动态阈值
            dynamic_threshold = self.static_capacity * beta_wind * energy_factor
            
            # --- C. 计算拥堵阻抗 (Cost) ---
            # 将“硬性的封锁”转化为“软性的高成本”
            density = pred_densities.get(s_id, 0)
            congestion_ratio = density / dynamic_threshold if dynamic_threshold > 0 else 10.0
            
            if congestion_ratio > 1.0:
                # 严重拥堵：指数级惩罚，迫使微观规划器绕路
                new_cost = self.config.BASE_COST * (congestion_ratio ** 4)
            elif congestion_ratio > 0.8:
                # 轻微拥堵：线性惩罚
                new_cost = self.config.BASE_COST * (1 + congestion_ratio)
            else:
                # 畅通
                new_cost = self.config.BASE_COST
            
            # --- D. 更新空域图权重 ---
            # 将扇区成本映射到所有进入该扇区的航路(Edge)上
            self.airspace.update_sector_cost(s_id, new_cost)
            
        return pred_densities # 用于日志记录

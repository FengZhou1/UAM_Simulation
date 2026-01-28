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
    def __init__(self, airspace, config, model_path=None):
        self.airspace = airspace
        self.config = config
        self.static_capacity = config.SECTOR_CAPACITY
        
        # ST-GAT 模型参数
        self.prediction_horizon = 5 # 预测未来 5 个时间步
        
        # Hardware / Model Setup
        self.model = None
        self.history_buffer = [] 
        self.max_history = 5 # Sequence length needed for model
        
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        
        if TORCH_AVAILABLE:
            # Use provided path or default if not specified but don't force 'stgat_model.pth' unless it exists
            # Logic: If model_path provided, use it. If None, try finding default. If neither, mode is "heuristic".
            # For DAgger warmup, we might pass model_path=None explicitly to ensure no model is loaded.
            
            if model_path:
                self.model_path = model_path
            else:
                self.model_path = 'stgat_model.pth' # Fallback default
            
            # Check if we should try loading
            if model_path is not None or os.path.exists(self.model_path):
                if os.path.exists(self.model_path):
                    try:
                        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                        
                        # Saved metadata
                        self.feat_mean = checkpoint.get('feat_mean', np.zeros(2))
                        self.feat_std = checkpoint.get('feat_std', np.ones(2))
                        self.saved_adj = checkpoint.get('adj', None)
                        if self.saved_adj is not None:
                             self.saved_adj = torch.FloatTensor(self.saved_adj).to(self.device)

                        # Initialize Model (Must match training Params)
                        # Input: Density, AvgSoc -> 2 features
                        self.model = STGAT(nfeat=2, nhid=64, nclass=1, dropout=0, alpha=0.2, nheads=4)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.to(self.device)
                        self.model.eval()
                        print(f"ST-GAT Model loaded from {self.model_path} on {self.device}")
                    except Exception as e:
                        print(f"Failed to load ST-GAT model: {e}")
                        self.model = None
            else:
                 print(f"Model file {self.model_path} not found. Running in Data Collection / Heuristic Mode.")
    
    def predict_sector_density(self, aircraft_list):
        """
        [Macro Layer 1]: 密度预测 (The Brain)
        模拟 ST-GAT 的输出：基于当前趋势预测未来密度。
        """
        # 1. 统计当前状态
        # We need a consistent ordering of nodes for the model
        sorted_nodes = sorted(list(self.airspace.graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        
        current_counts = np.zeros(len(sorted_nodes))
        current_socs = np.zeros(len(sorted_nodes))
        
        # Temp accumulators
        temp_counts = {n: 0 for n in sorted_nodes}
        temp_socs = {n: [] for n in sorted_nodes}
        
        for ac in aircraft_list:
            if not ac.finished:
                rid = ac.current_region_id
                if rid in temp_counts:
                    temp_counts[rid] += 1
                    temp_socs[rid].append(ac.battery_soc)
        
        for i, rid in enumerate(sorted_nodes):
            current_counts[i] = temp_counts[rid]
            socs = temp_socs[rid]
            current_socs[i] = sum(socs)/len(socs) if socs else 1.0
            
        # Update History for Model
        current_features = np.stack([current_counts, current_socs], axis=1) # (N, 2)
        self.history_buffer.append(current_features)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        # 2. 模拟 ST-GAT 预测 (加入惯性和波动) OR Real Inference
        pred_densities = {}
        sector_states = {} # 存储用于计算阈值的状态
        
        use_model = (self.model is not None) and (len(self.history_buffer) == self.max_history) and TORCH_AVAILABLE

        if use_model:
            with torch.no_grad():
                # Prepare Input: (1, T, N, F)
                x_seq = np.array(self.history_buffer)
                # Normalize
                x_seq = (x_seq - self.feat_mean) / (self.feat_std + 1e-5)
                x_tensor = torch.FloatTensor(x_seq).unsqueeze(0).to(self.device)
                
                # Adj
                adj = self.saved_adj if self.saved_adj is not None else torch.eye(len(sorted_nodes)).to(self.device)
                
                # Forward
                # STGAT model expects (Batch, Seq, Nodes, Feat)
                out = self.model(x_tensor, adj) # (1, N, 1)
                
                pred_raw = out.squeeze().cpu().numpy()
                # Denormalize (Density is feature 0)
                pred_raw = pred_raw * (self.feat_std[0] + 1e-5) + self.feat_mean[0]
                
                for i, rid in enumerate(sorted_nodes):
                    pred_densities[rid] = max(0, pred_raw[i])
                    sector_states[rid] = {'soc': current_socs[i]}
        else:
            # Fallback / Simulation Logic (Prompy Methodology)
            for i, s_id in enumerate(sorted_nodes):
                # 简单模拟：假设未来密度会受当前流入趋势影响 (x 1.1) 加上随机波动
                pred = current_counts[i] * 1.1 + np.random.normal(0, 0.5)
                pred_densities[s_id] = max(0, pred)
                
                # 计算该扇区机群的平均电量 (用于下一步的能量豁免)
                sector_states[s_id] = {'soc': current_socs[i]}
            
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
            # Prompt code uses -10 * avg_soc
            energy_factor = 1.0 + lambda_energy * math.exp(-10 * avg_soc)
            
            # 3. 最终动态阈值
            dynamic_threshold = self.static_capacity * beta_wind * energy_factor
            
            # --- C. 计算拥堵阻抗 (Cost) ---
            # 将“硬性的封锁”转化为“软性的高成本”
            density = pred_densities.get(s_id, 0)
            
            # Avoid division by zero
            denom = dynamic_threshold if dynamic_threshold > 0.1 else 0.1
            congestion_ratio = density / denom
            
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

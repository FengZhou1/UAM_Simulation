import numpy as np
import math

class STGATController:
    def __init__(self, airspace, config):
        self.airspace = airspace
        self.config = config
        self.static_capacity = config.SECTOR_CAPACITY
        
        # ST-GAT 模型参数 (这里用简单的逻辑模拟预测，实际部署时替换为 torch model)
        self.prediction_horizon = 5 # 预测未来 5 个时间步
    
    def predict_sector_density(self, aircraft_list):
        """
        [Macro Layer 1]: 密度预测 (The Brain)
        模拟 ST-GAT 的输出：基于当前趋势预测未来密度。
        """
        # 1. 统计当前密度
        current_counts = {s_id: 0 for s_id in self.airspace.graph.nodes()} # Use graph nodes as sector/region IDs
        avg_soc = {s_id: [] for s_id in self.airspace.graph.nodes()}
        
        for ac in aircraft_list:
            if not ac.finished:
                # Use ac.current_region_id instead of get_sector_id based on position if available
                # Or use airspace.get_sector_id(ac.pos)
                current_sector = ac.current_region_id
                if current_sector in current_counts:
                    current_counts[current_sector] += 1
                    avg_soc[current_sector].append(ac.battery_soc)
        
        # 2. 模拟 ST-GAT 预测 (加入惯性和波动)
        # 在论文中，这里是: pred_y = Model(history_x, adj_matrix)
        pred_densities = {}
        sector_states = {} # 存储用于计算阈值的状态
        
        for s_id in self.airspace.graph.nodes():
            # 简单模拟：假设未来密度会受当前流入趋势影响 (x 1.1) 加上随机波动
            count = current_counts.get(s_id, 0)
            pred = count * 1.1 + np.random.normal(0, 1)
            pred_densities[s_id] = max(0, pred)
            
            # 计算该扇区机群的平均电量 (用于下一步的能量豁免)
            socs = avg_soc.get(s_id, [])
            mean_soc = sum(socs)/len(socs) if socs else 1.0
            sector_states[s_id] = {'soc': mean_soc}
            
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

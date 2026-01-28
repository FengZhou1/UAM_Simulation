import numpy as np

class Config:
    # --- 仿真参数 ---
    DT = 0.1                # 时间步长 (s)
    MAX_TIME = 3600         # 最大仿真时长 (s)
    
    # --- 空域参数 ---
    REGION_RADIUS = 250.0   # 区域半径 (m)
    LAYER_HEIGHTS = [400, 600] # 层高度 (m)
    
    # --- 航空器参数 ---
    DETECTION_RADIUS = 200.0 # 感知半径 (m)
    SAFETY_RADIUS = 50.0     # 安全半径 (m)
    MAX_SPEED = 20.0         # 最大速度 (m/s)
    CONTROL_GAIN = 5.0       # 控制增益 l [-]
    
    # --- 路径规划参数 (MFD & Algo 1) ---
    MAX_CANDIDATE_PATHS = 6  # 每个OD对的最大候选路径数
    
    # MFD 拥堵控制参数 (Eq. 3)
    # 这些参数通常通过拟合得到，这里使用经验值近似
    N_CR = 10.0              # 临界累积量 (Critical Accumulation)
    
    @staticmethod
    def get_congestion_speed(n_aircraft):
        """
        论文公式 (3): 基于区域内飞机数量计算当前通行速度
        V = V_max * exp(-N + N_cr) / (1 + exp(-N + N_cr))
        """
        exponent = -n_aircraft + Config.N_CR
        # 避免溢出
        exponent = np.clip(exponent, -100, 100)
        factor = np.exp(exponent) / (1.0 + np.exp(exponent))
        return Config.MAX_SPEED * factor
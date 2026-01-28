# utils.py
import numpy as np
import math

def dist(p1, p2):
    """计算3D欧几里得距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def generate_hex_grid_centers(radius, layers, grid_size=5):
    """
    生成六边形网格中心点坐标
    使用 Axial Coordinates (q, r) 转换为 (x, y)
    """
    regions = {}
    region_id_counter = 0
    
    width = math.sqrt(3) * radius
    height = 2 * radius
    
    for z in layers:
        for q in range(-grid_size, grid_size + 1):
            for r in range(-grid_size, grid_size + 1):
                # 六边形距离约束，保持形状接近圆形/六边形
                if abs(q + r) <= grid_size:
                    x = width * (q + r/2)
                    y = height * (3/4) * r
                    
                    # Store: {ID: (x, y, z)}
                    regions[region_id_counter] = np.array([x, y, z])
                    region_id_counter += 1
    return regions

def get_closest_region(pos, regions):
    """找到坐标点 pos 归属的 Region ID"""
    min_d = float('inf')
    closest_id = -1
    for rid, center in regions.items():
        # 简单处理：仅基于XY平面距离判断区域归属，忽略高度层差异
        # 实际应用中应先匹配Z层，再匹配XY
        d = dist(pos, center)
        if d < min_d:
            min_d = d
            closest_id = rid
    return closest_id
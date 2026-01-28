# airspace.py
import networkx as nx
import numpy as np
from utils import generate_hex_grid_centers, dist
from config import Config

class Airspace:
    def __init__(self):
        self.regions = generate_hex_grid_centers(Config.REGION_RADIUS, Config.LAYER_HEIGHTS)
        self.graph = self._build_graph()
        self.region_occupancy = {rid: 0 for rid in self.regions} # N(R_l)
        
    def _build_graph(self):
        """
        构建 NetworkX 有向图
        节点：Region ID
        边：相邻区域（距离阈值内）
        """
        G = nx.DiGraph()
        
        # 添加节点
        for rid, pos in self.regions.items():
            G.add_node(rid, pos=pos)
            
        # 添加边 (水平邻居 + 垂直管道)
        neighbor_dist_limit = Config.REGION_RADIUS * 2.1 # 略大于2倍半径
        
        ids = list(self.regions.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                d = dist(self.regions[id1], self.regions[id2])
                
                # 判定相邻：如果是同层且距离合适，或不同层但在同一XY位置(垂直管道)
                is_same_layer = abs(self.regions[id1][2] - self.regions[id2][2]) < 1.0
                is_vertical_aligned = dist(self.regions[id1][:2], self.regions[id2][:2]) < 10.0
                
                if (is_same_layer and d <= neighbor_dist_limit) or \
                   (not is_same_layer and is_vertical_aligned):
                    # 双向连接
                    G.add_edge(id1, id2, weight=d)
                    G.add_edge(id2, id1, weight=d)
        return G

    def update_occupancy(self, aircraft_list):
        """更新每个区域当前的飞机数量 N(R_l)"""
        for rid in self.region_occupancy:
            self.region_occupancy[rid] = 0
            
        for ac in aircraft_list:
            if not ac.finished:
                rid = ac.current_region_id
                if rid in self.region_occupancy:
                    self.region_occupancy[rid] += 1
# route_guidance.py
import networkx as nx
import numpy as np
import logging
from itertools import islice
from config import Config

logger = logging.getLogger(__name__)

class RouteGuidance:
    def __init__(self, airspace):
        self.airspace = airspace
        self.candidate_paths_cache = {} # 缓存候选路径
        
    def get_candidate_paths(self, origin_rid, dest_rid):
        """
        复现 Algorithm 1: 候选路径构建
        使用 NetworkX 的简单路径搜索作为近似
        """
        key = (origin_rid, dest_rid)
        if key in self.candidate_paths_cache:
            return self.candidate_paths_cache[key]
        
        # 搜索 k 条最短路径
        try:
            # 简化：使用 k-shortest paths
            paths = list(islice(nx.shortest_simple_paths(
                self.airspace.graph, origin_rid, dest_rid, weight='weight'), 
                Config.MAX_CANDIDATE_PATHS))
        except nx.NetworkXNoPath:
            paths = []
            
        self.candidate_paths_cache[key] = paths
        return paths

    def estimate_path_cost(self, path, current_occupancy):
        """
        复现 Algorithm 3 的核心思想 (Fast Cost Estimation)
        基于当前拥堵状况估算路径耗时
        """
        total_time = 0
        
        # 这是一个简化版的估算，论文中使用矩阵同步更新，
        # 这里我们假设当前拥堵状态在短时间内不变（这是常用的近似）
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # 获取路段长度
            distance = self.airspace.graph[u][v]['weight']
            
            # 获取进入区域 u 的拥堵程度
            n_aircraft = current_occupancy.get(u, 0)
            
            # 计算受拥堵影响的速度 (Eq. 3)
            speed = Config.get_congestion_speed(n_aircraft)
            
            travel_time = distance / max(speed, 0.1) # 防止除零
            total_time += travel_time
            
        return total_time

    def solve_approximate_optimal_paths(self, active_aircraft):
        """
        复现 Algorithm 4: 增量式近似最优搜索
        """
        # 1. 排序：按 OD 距离（直线距离）从小到大排序
        # 论文逻辑：短途优先，长途让路
        active_aircraft.sort(key=lambda ac: np.linalg.norm(ac.pos - ac.destination))
        
        # 临时拥堵表，用于在规划过程中累加预测的拥堵
        temp_occupancy = self.airspace.region_occupancy.copy()
        
        for ac in active_aircraft:
            # 找到起点和终点的 Region ID
            start_rid = ac.current_region_id
            # 简单处理：如果找不到对应的 Region，找最近的
            # (实际应在 Aircraft 类中维护 origin_rid, dest_rid)
            from utils import get_closest_region
            dest_rid = get_closest_region(ac.destination, self.airspace.regions)
            
            if start_rid == -1: continue # 尚未进入空域
            
            # FIX: 如果已经在终点所在的区域，就不需要再规划路径了，直接飞向终点坐标即可
            # 避免被拉回区域中心
            if start_rid == dest_rid:
                ac.path_waypoints = []
                ac.current_wp_index = 0
                continue

            candidates = self.get_candidate_paths(start_rid, dest_rid)
            if not candidates: continue
            
            best_path = None
            min_cost = float('inf')
            
            # 2. 遍历候选路径，选择当前代价最小的
            for path in candidates:
                cost = self.estimate_path_cost(path, temp_occupancy)
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
            
            # 3. 分配路径并更新临时拥堵表
            if best_path:
                # 将路径上的 Region 坐标转换为 Waypoints
                waypoints = [self.airspace.regions[rid] for rid in best_path]
                
                # FIX: 如果路径包含当前节点，且长度 > 1，则跳过第一个节点（当前节点），直接飞向下一个
                # 避免每次重规划都飞回当前区域中心，导致“来回摇摆”
                if len(best_path) > 1 and best_path[0] == start_rid:
                    waypoints = waypoints[1:]
                
                ac.path_waypoints = waypoints
                ac.current_wp_index = 0 # 重置航点索引
                
                logger.info(f"Assigned path to Aircraft {ac.id}: {best_path} with cost {min_cost}")

                # 更新预测拥堵 (简单地给路径上的区域加权，论文有更复杂的矩阵法)
                for rid in best_path:
                    temp_occupancy[rid] += 1
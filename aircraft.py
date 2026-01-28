# aircraft.py
import numpy as np
import logging
from config import Config
import networkx as nx
from utils import get_closest_region, dist
from collision_avoidance import CollisionAvoidance

logger = logging.getLogger(__name__)

class Aircraft:
    def __init__(self, id, origin, destination, start_time, config):
        self.id = id
        self.pos = np.array(origin, dtype=float)
        self.velocity = np.zeros(3)
        self.destination = np.array(destination, dtype=float)
        self.config = config # Add config
        
        self.path = [] # Node IDs
        self.path_waypoints = [] # Coords
        self.current_wp_index = 0
        
        self.start_time = start_time
        self.finished = False
        self.finish_time = 0.0
        self.current_region_id = -1
        
        # 对应 Eq. (7) 的 omega
        self.omega = Config.CONTROL_GAIN * Config.DT
        
        self.battery_soc = 1.0
        self.reroute_timer = 0
        self.destination_node = -1 # Will be set on first plan

    def get_target_waypoint(self):
        """获取当前导航目标点"""
        if self.path_waypoints and self.current_wp_index < len(self.path_waypoints):
            return self.path_waypoints[self.current_wp_index]
        return self.destination

    def update(self, dt, airspace, neighbors=None, current_time=0.0, routing_mode='stgat'):
        """
        Modified update loop integrating Physics, Battery and Micro-Path Planning
        """
        if self.finished: return

        # 1. 电池消耗
        speed = np.linalg.norm(self.velocity)
        if speed < 1.0: # 悬停
            self.battery_soc -= self.config.HOVER_DISCHARGE_RATE * dt
        else:
            self.battery_soc -= self.config.CRUISE_DISCHARGE_RATE * dt
        
        # Cap SoC
        if self.battery_soc < 0: 
            self.battery_soc = 0
            # Battery depleted, force finish (crash/land)
            self.finished = True
            self.finish_time = current_time # Record time of death
            logger.info(f"Aircraft {self.id} depleted battery at {current_time:.2f}s.")
            return

        # Timeout Check (防止仿真卡死)
        # 假设最大允许时间是预计时间的 3 倍或者固定值
        # 增加超时限制，因为4架飞机卡了很久
        if (current_time - self.start_time) > self.config.MAX_TIME * 0.9 and current_time > 0:
             self.finished = True
             self.finish_time = current_time # Record timeout time
             logger.info(f"Aircraft {self.id} timed out at {current_time:.2f}s.")
             return

        # Determine current region
        self.current_region_id = get_closest_region(self.pos, airspace.regions)
        
        # Deadlock/Stuck check
        # 如果速度很慢且持续很长时间，尝试强制重规划或某种解脱措施
        speed = np.linalg.norm(self.velocity)
        if speed < 1.0 and not self.finished and neighbors:
             # 可能卡在拥堵或死锁中
             self.reroute_timer += dt * 5 # 加速重规划计时器
        
        # Initialize destination node if not set
        if self.destination_node == -1:
             self.destination_node = get_closest_region(self.destination, airspace.regions)
             self.plan_path(airspace, mode=routing_mode) # Initial plan

        # 3. [Micro Layer]: 周期性重规划
        self.reroute_timer += dt
        # First plan or periodic replan
        if not self.path or self.reroute_timer > self.config.REROUTE_INTERVAL:
            self.plan_path(airspace, mode=routing_mode)
            self.reroute_timer = 0

        # 2. 路径执行 & 避障 (Calculate v_command)
        if neighbors is None:
            neighbors = []
            
        # Use CollisionAvoidance module to compute velocity
        # This encapsulates the "Guidance" (Preferred Velocity) + "avoidance"
        v_command = CollisionAvoidance.compute_velocity_command(self, neighbors)
            
        # Call physics update
        self.update_state(v_command, current_time)

    def plan_path(self, airspace, mode='static'):
        """
        使用 Dijkstra/A* 基于配置模式进行规划
        Modes:
        - 'static': 静态最短路 (Weight = distance)
        - 'dynamic': 动态反馈 (Weight = distance + congestion cost)
        - 'stochastic': 随机扰动 (K-shortest)
        - 'stgat': (Default inference) use macro-controlled weights
        """
        try:
            current_node = self.current_region_id
            if current_node == -1:
                return # Not in airspace yet

            # 区分训练生成模式和推理模式
            # 如果是 data_collection，可以由外部传入 mode，或者根据配置随机选择
            if mode == 'static':
                new_path = nx.shortest_path(
                    airspace.graph, 
                    source=current_node, 
                    target=self.destination_node, 
                    weight='static_dist'
                )
            elif mode == 'dynamic':
                # 在 simulation.py 中，我们已经根据拥堵更新了 'weight'
                # 这里假设 'weight' 已经是 dynamic cost
                new_path = nx.shortest_path(
                    airspace.graph, 
                    source=current_node, 
                    target=self.destination_node, 
                    weight='weight' 
                )
            elif mode == 'stochastic':
                # K-Shortest simple paths is very slow for large graphs.
                # Simplified Stochastic: Add random noise to weights
                # Create a temporary view or copy is too expensive.
                # Alternative: A* with randomized heuristic or randomized edge weights
                # Let's use a simple randomized strategy: 
                # With 20% chance, pick a random neighbor as next step, then shortest path
                if np.random.random() < 0.2:
                    neighbors = list(airspace.graph.neighbors(current_node))
                    if neighbors:
                        next_hop = np.random.choice(neighbors)
                        remaining_path = nx.shortest_path(
                            airspace.graph,
                            source=next_hop,
                            target=self.destination_node,
                            weight='static_dist'
                        )
                        new_path = [current_node] + remaining_path
                    else:
                        new_path = nx.shortest_path(airspace.graph, source=current_node, target=self.destination_node, weight='static_dist')
                else:
                    new_path = nx.shortest_path(airspace.graph, source=current_node, target=self.destination_node, weight='static_dist')

            else: # Default 'stgat' or standard usage
                 new_path = nx.shortest_path(
                    airspace.graph, 
                    source=current_node, 
                    target=self.destination_node, 
                    weight='weight'
                )

            self.path = new_path
            # Convert path (IDs) to waypoints (Coords)
            self.path_waypoints = [airspace.regions[rid] for rid in self.path]
            self.current_wp_index = 1 if len(self.path_waypoints) > 1 else 0
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # 无路可走，保持原地悬停
            self.path = []
            self.path_waypoints = []
            
    def update_state(self, v_command, current_time=0.0):
        """
        动力学更新，复现 Eq. (7) 和 Eq. (5a)
        v(k+1) = (1 - omega) * v(k) + omega * v_command
        p(k+1) = p(k) + v(k) * dt
        """
        if self.finished: return

        # 速度更新 (Eq. 7)
        self.velocity = (1 - self.omega) * self.velocity + self.omega * v_command
        
        # Clamp Speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.config.MAX_SPEED:
            self.velocity = (self.velocity / speed) * self.config.MAX_SPEED
            
        # 位置更新
        self.pos += self.velocity * self.config.DT
        
        # 检查是否到达终点
        dist_to_dest = np.linalg.norm(self.pos - self.destination)
        if dist_to_dest < self.config.DETECTION_RADIUS: # Arrived
            self.finished = True
            self.finish_time = current_time
            logger.info(f"Aircraft {self.id} finished at {current_time:.2f}s")
        
        # 位置更新 (Eq. 5a)
        self.pos += self.velocity * Config.DT
        
        # 检查是否到达当前航点
        target = self.get_target_waypoint()
        if dist(self.pos, target) < 20.0: # 到达判断阈值
            # logger.info(f"Aircraft {self.id} reached waypoint {self.current_wp_index}: {target}")
            self.current_wp_index += 1
            
        # 检查是否到达终点
        if dist(self.pos, self.destination) < 10.0:
            # logger.info(f"Aircraft {self.id} reached destination.")
            self.finished = True
            self.finish_time = current_time
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

    def update(self, dt, airspace, neighbors=None):
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
        if self.battery_soc < 0: self.battery_soc = 0

        # Determine current region
        self.current_region_id = get_closest_region(self.pos, airspace.regions)
        
        # Initialize destination node if not set
        if self.destination_node == -1:
             self.destination_node = get_closest_region(self.destination, airspace.regions)

        # 3. [Micro Layer]: 周期性重规划
        self.reroute_timer += dt
        # First plan or periodic replan
        if not self.path or self.reroute_timer > self.config.REROUTE_INTERVAL:
            self.plan_path(airspace)
            self.reroute_timer = 0

        # 2. 路径执行 & 避障 (Calculate v_command)
        if neighbors is None:
            neighbors = []
            
        # Use CollisionAvoidance module to compute velocity
        # This encapsulates the "Guidance" (Preferred Velocity) + "avoidance"
        v_command = CollisionAvoidance.compute_velocity_command(self, neighbors)
            
        # Call physics update
        self.update_state(v_command)

    def plan_path(self, airspace):
        """
        使用 Dijkstra/A* 基于当前宏观层发布的 weight 进行规划
        """
        try:
            current_node = self.current_region_id
            if current_node == -1:
                current_node = get_closest_region(self.pos, airspace.regions)
                
            new_path = nx.shortest_path(
                airspace.graph, 
                source=current_node, 
                target=self.destination_node, 
                weight='weight' # 关键：使用动态权重
            )
            self.path = new_path
            # Convert path (IDs) to waypoints (Coords)
            # Only update waypoints if path changed significantly? 
            # For simplicity, overwrite.
            # But we need to be careful about current_wp_index.
            # Since new path starts from current_node, we can reset index to 1 (next node). 0 is current.
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
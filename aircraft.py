# aircraft.py
import numpy as np
import logging
from config import Config
from utils import dist

logger = logging.getLogger(__name__)

class Aircraft:
    def __init__(self, id, origin, destination, start_time):
        self.id = id
        self.pos = np.array(origin, dtype=float)
        self.velocity = np.zeros(3)
        self.destination = np.array(destination, dtype=float)
        
        self.path_waypoints = [] # 路径规划给出的航点 (Region Centers)
        self.current_wp_index = 0
        
        self.start_time = start_time
        self.finished = False
        self.finish_time = 0.0
        self.current_region_id = -1
        
        # 对应 Eq. (7) 的 omega
        self.omega = Config.CONTROL_GAIN * Config.DT
        
    def get_target_waypoint(self):
        """获取当前导航目标点"""
        if self.current_wp_index < len(self.path_waypoints):
            return self.path_waypoints[self.current_wp_index]
        return self.destination

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
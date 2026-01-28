# collision_avoidance.py
import numpy as np
import logging
from config import Config
from utils import dist

logger = logging.getLogger(__name__)

class CollisionAvoidance:
    @staticmethod
    def compute_velocity_command(agent, neighbors):
        """
        计算避障后的速度指令 v_c
        基于 Velocity Obstacle (VO) 的采样简化版
        对应 Eq (9): min || v - v_preferred || subject to v in SafeSet
        """
        
        # 1. 计算首选速度 (Preferred Velocity) Eq. (8)
        target_pos = agent.get_target_waypoint()
        direction = target_pos - agent.pos
        dist_to_target = np.linalg.norm(direction)
        
        if dist_to_target > 0.1:
            # 接近目标时减速，防止超调和震荡
            # 简单的 P 控制：速度与距离成正比，上限为 MAX_SPEED
            speed = min(Config.MAX_SPEED, dist_to_target * 1.0) 
            v_pref = (direction / dist_to_target) * speed
        else:
            v_pref = np.zeros(3)
            
        if not neighbors:
            return v_pref
            
        # 2. 简单的 VO 采样求解
        # 在 v_pref 周围采样，找到不与任何邻居碰撞且最接近 v_pref 的速度
        best_v = np.zeros(3)
        min_deviation = float('inf')
        
        # 采样参数
        sample_angles = np.linspace(0, 2*np.pi, 16) # 水平采样
        sample_speeds = [Config.MAX_SPEED, Config.MAX_SPEED * 0.5, 0]
        
        # 添加 v_pref 本身作为首选检查
        candidate_velocities = [v_pref]
        
        # 生成候选速度 (简化的3D采样，主要关注XY平面避障，Z轴简单处理)
        for s in sample_speeds:
            for angle in sample_angles:
                vx = s * np.cos(angle)
                vy = s * np.sin(angle)
                vz = v_pref[2] # 保持垂直意图
                candidate_velocities.append(np.array([vx, vy, vz]))
                
        for v_cand in candidate_velocities:
            collision = False
            for other in neighbors:
                # 检查 v_cand 是否会导致未来 collision_time 内碰撞
                # 相对位置
                p_rel = other.pos - agent.pos
                # 相对速度
                v_rel = v_cand - other.velocity 
                
                # 碰撞检测逻辑 (VO Core)
                # 求解 ray-sphere intersection
                # p(t) = p_rel + v_rel * t
                # check if ||p(t)|| < 2 * r_s for t in [0, tau]
                
                # 简化判定：如果相对速度指向对方，且距离足够近
                dist_rel = np.linalg.norm(p_rel)
                if dist_rel > Config.DETECTION_RADIUS:
                    continue
                    
                # 投影 v_rel 到 p_rel
                dot = np.dot(v_rel, p_rel)
                if dot > 0: # 正在接近 (Fix: dot > 0 means approaching)
                    # 估算最近点距离
                    t_closest = dot / (np.linalg.norm(v_rel)**2 + 1e-6) # Fix: t_closest calculation sign
                    closest_p = p_rel - v_rel * t_closest # Fix: vector math
                    min_dist = np.linalg.norm(closest_p)
                    
                    if min_dist < 2 * Config.SAFETY_RADIUS:
                        collision = True
                        break
            
            if not collision:
                # 计算与 v_pref 的偏差
                deviation = np.linalg.norm(v_cand - v_pref)
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_v = v_cand

        if np.linalg.norm(best_v - v_pref) > 0.1:
             logger.info(f"Agent {agent.id} avoiding collision. Deviation: {min_deviation:.2f}")

        return best_v
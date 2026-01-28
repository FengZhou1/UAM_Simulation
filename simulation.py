# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import RegularPolygon
import logging
import networkx as nx 
from config import Config
from utils import get_closest_region
from airspace import Airspace
from aircraft import Aircraft
# from route_guidance import RouteGuidance # Removed
# from collision_avoidance import CollisionAvoidance # Removed in favor of internal update
from macro_control import STGATController

class WindField:
    def get_wind_at(self, pos):
        # Return scalar wind speed
        return 5.0

def run_simulation(mode='guided', visualize=True, routing_policy='stgat', model_path=None, save_data=False, output_data='traffic_data.npy'):
    # 配置日志
    logger = logging.getLogger('Simulation')
    # Clearing handlers to avoid duplicates in iterative runs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.basicConfig(
        filename='simulation.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )
    
    logger.info(f"Starting Simulation in {mode} mode with policy {routing_policy}...")
    print(f"Starting Simulation in {mode} mode with policy {routing_policy}. Model: {model_path}")

    # 1. 初始化 (Initialize)
    logger.info("Initializing Airspace...")
    print("Initializing Airspace...")
    airspace = Airspace()
    # rg_module = RouteGuidance(airspace) # Removed
    
    # Init Macro Controller
    # Pass model_path explicitly
    macro_controller = STGATController(airspace, Config, model_path=model_path)
    wind_field = WindField()
    
    aircraft_list = []
    
    # 场景生成：两组主要航线
    # Group 1: 左 -> 右 (20架)
    for i in range(20):
        start = [np.random.uniform(-1500, -1100), np.random.uniform(-400, 400), 400]
        end = [np.random.uniform(1100, 1500), np.random.uniform(-400, 400), 400]
        # Update Aircraft init with Config
        aircraft_list.append(Aircraft(i+1, start, end, 0, Config))
        
    # Group 2: 上 -> 下 (20架)
    for i in range(20):
        start = [np.random.uniform(-400, 400), np.random.uniform(1100, 1500), 400]
        end = [np.random.uniform(-400, 400), np.random.uniform(-1500, -1100), 400]
        aircraft_list.append(Aircraft(i+21, start, end, 0, Config))

    # 数据记录
    history_pos = {ac.id: [] for ac in aircraft_list}
    
    logger.info(f"Starting Simulation Loop with {len(aircraft_list)} aircraft...")
    print("Starting Simulation Loop...")

    # --- 初始化实时绘图 ---
    region_patches = {}
    ac_scatters = {}
    ac_trails = {}
    ac_targets = {}

    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111) # 2D 视图
        
        # 计算显示范围 (基于起点和终点)
        all_points = []
        for ac in aircraft_list:
            all_points.append(ac.pos)
            all_points.append(ac.destination)
        all_points = np.array(all_points)
        margin = 100
        min_bound = np.min(all_points, axis=0) - margin
        max_bound = np.max(all_points, axis=0) + margin
        
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        # ax.set_zlim(min_bound[2], max_bound[2]) # 2D 不需要 Z 轴限制
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Altitude [m]')
        ax.set_title(f'UAM Simulation ({mode.capitalize()} Mode)')
        ax.set_aspect('equal') # 保持比例一致
        
        # 绘制背景节点 (只绘制范围内的)
        shown_rids = set()
        
        for rid, pos in airspace.regions.items():
            if (pos >= min_bound).all() and (pos <= max_bound).all():
                shown_rids.add(rid)
                
                # Pointy topped hexagon
                hex_patch = RegularPolygon((pos[0], pos[1]), numVertices=6, radius=Config.REGION_RADIUS, 
                                           orientation=0, 
                                           alpha=0.1, edgecolor='gray', facecolor='none', linewidth=0.8)
                ax.add_patch(hex_patch)
                region_patches[rid] = hex_patch
                
                # 画节点中心 (2D)
                ax.scatter(pos[0], pos[1], c='gray', marker='+', s=20, alpha=0.5)

        # 画边 (连接关系) (2D)
        for u, v in airspace.graph.edges():
            if u in shown_rids and v in shown_rids:
                p1 = airspace.regions[u]
                p2 = airspace.regions[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        c='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)

        # 初始化飞机绘图对象 (2D)
        
        # 选择要显示轨迹的飞机 ID (每组选几架)
        # Group 1 (1-20): 选 1, 5, 10
        # Group 2 (21-40): 选 21, 25, 30
        trace_ids = [1, 5, 10, 21, 25, 30]
        
        colors = plt.cm.jet(np.linspace(0, 1, len(aircraft_list)))
        for i, ac in enumerate(aircraft_list):
            # 飞机图标
            ac_scatters[ac.id] = ax.scatter([ac.pos[0]], [ac.pos[1]], 
                                          c=[colors[i]], s=50, marker='^')
            
            # 起点 (绿色圆点)
            ax.scatter(ac.pos[0], ac.pos[1], c='green', marker='o', s=20, alpha=0.5)
            # 终点 (红色X)
            ax.scatter(ac.destination[0], ac.destination[1], c='red', marker='x', s=30, alpha=0.6)
            
            # 初始化轨迹线 (仅部分)
            if ac.id in trace_ids:
                trail, = ax.plot([], [], c=colors[i], linewidth=1, alpha=0.6)
                ac_trails[ac.id] = trail
                
                # 目标点 (星号)
                target_scatter = ax.scatter([], [], c=[colors[i]], marker='*', s=100, edgecolors='black', linewidths=0.5)
                ac_targets[ac.id] = target_scatter
        
        print("Opening visualization window...")
        plt.pause(0.5) # 给窗口一点时间弹出

    # 2. 时间循环 (Time Loop)
    t = 0
    max_steps = int(Config.MAX_TIME / Config.DT)
    macro_update_steps = int(Config.MACRO_INTERVAL / Config.DT)
    
    # --- Data Collection Buffer ---
    data_buffer = [] # To store list of (Density, SoC) vectors
    
    while t < max_steps: # 运行直到完成或超时
        current_time = t * Config.DT
        t += 1
        
        # --- A. 感知 (Perceive) ---
        active_aircraft = [ac for ac in aircraft_list if not ac.finished]
        if not active_aircraft:
            logger.info("All aircraft finished.")
            print("All aircraft finished.")
            break
            
        # 更新空域拥堵统计
        airspace.update_occupancy(active_aircraft)
        # ac.current_region_id is updated inside ac.update() now
        
        # --- Data Collection Logic ---
        # Collect data every 1 second (10 steps) to ensure sufficient training data
        collection_interval = int(1.0 / Config.DT)
        if save_data and t % collection_interval == 0:
            sorted_nodes = sorted(list(airspace.graph.nodes()))
            current_counts = np.zeros(len(sorted_nodes))
            current_socs = np.zeros(len(sorted_nodes))
            temp_counts = {n: 0 for n in sorted_nodes}
            temp_socs = {n: [] for n in sorted_nodes}
             
            for ac in active_aircraft:
                 rid = ac.current_region_id
                 if rid in temp_counts:
                     temp_counts[rid] += 1
                     temp_socs[rid].append(ac.battery_soc)
            
            for i, rid in enumerate(sorted_nodes):
                current_counts[i] = temp_counts[rid]
                socs = temp_socs[rid]
                current_socs[i] = sum(socs)/len(socs) if socs else 1.0
            
            # Feature vector: [Density, SoC]
            frame_feats = np.stack([current_counts, current_socs], axis=1) # (N, 2)
            data_buffer.append(frame_feats)
            # if len(data_buffer) % 10 == 0:
            #    print(f"Collected {len(data_buffer)} frames of data...")
        
        # --- 1. 宏观控制层 (Macro) ---
        if t % macro_update_steps == 0:
            pred_map = macro_controller.update_airspace_costs(active_aircraft, wind_field)
            # print(f"[Time {current_time}] Macro Control Update.")

        # --- 2. 微观执行层 (Micro) ---
        for ac in active_aircraft:
            # 寻找邻居 (Simple pairwise check)
            neighbors = []
            for other in active_aircraft:
                if ac.id != other.id and np.linalg.norm(ac.pos - other.pos) < Config.DETECTION_RADIUS:
                    neighbors.append(other)
            
            ac.update(Config.DT, airspace, neighbors)
            # Pass current_time to update_state inside update is tricky because update() signature
            # Let's just update finish_time logic inside update_state, but we need to pass time.
            # Modified aircraft.py's update to call update_state with time
            
            # Actually, let's just make sure aircraft.update() passes current_time to update_state if needed
            # But currently aircraft.py update() doesn't take current_time.
            # We can just manually check finish here as a safeguard or trust update()'s logic if we fix it.
            # Let's fix aircraft.py signature in next step if needed, or pass it via update invocation
            
            # FIX: We need to pass current_time to ac.update for the timeout logic we added
            ac.update(Config.DT, airspace, neighbors, current_time=current_time)
            
            history_pos[ac.id].append(ac.pos.copy())
            
        if t % 200 == 0:
            print(f"Time Step: {t}, Active Aircraft: {len(active_aircraft)}")

            # Deadlock breaker: If active aircraft count is stagnant for too long?
            # Or just rely on aircraft-level timeout implemented in aircraft.py

        # --- 实时更新绘图 ---
        if visualize:
            # 每 5 步更新一次画面 (0.5s)
            if t % 5 == 0:
                try:
                    # 更新区域颜色 (Occupancy)
                    for rid, patch in region_patches.items():
                        occupancy = airspace.region_occupancy.get(rid, 0)
                        if occupancy > 0:
                            # 简单的颜色映射: 1->Green, 2->Yellow, 3+->Red
                            if occupancy <= 2:
                                patch.set_facecolor('green')
                                patch.set_alpha(0.3)
                            elif occupancy <= 4:
                                patch.set_facecolor('yellow')
                                patch.set_alpha(0.4)
                            else:
                                patch.set_facecolor('red')
                                patch.set_alpha(0.5)
                        else:
                            patch.set_facecolor('none')
                            patch.set_alpha(0.1)

                    for ac in aircraft_list:
                        if not ac.finished:
                            # 2D 更新: set_offsets 需要一个 (N, 2) 的数组
                            ac_scatters[ac.id].set_offsets(np.c_[ac.pos[0], ac.pos[1]])
                        
                        # 更新轨迹 (仅部分)
                        if ac.id in ac_trails:
                            hist = np.array(history_pos[ac.id])
                            if len(hist) > 1:
                                ac_trails[ac.id].set_data(hist[:, 0], hist[:, 1])
                            
                            # 更新实时目标点
                            target = ac.get_target_waypoint()
                            ac_targets[ac.id].set_offsets(np.c_[target[0], target[1]])

                    plt.draw()
                    plt.pause(0.05) # 增加暂停时间，让渲染更平滑可见
                except Exception as e:
                    logger.error(f"Plotting error: {e}")
            else:
                # 在不绘图的帧，也刷新一下事件循环，防止窗口卡死
                if t % 2 == 0: # 降低刷新频率
                    fig.canvas.flush_events() 
                # ac_targets[ac.id].set_offsets(np.c_[target[0], target[1]])



    # 3. 结果可视化
    print("Simulation finished.")
    
    # 统计信息
    finished_aircraft = [ac for ac in aircraft_list if ac.finished]
    avg_time = 0
    if finished_aircraft:
        avg_time = sum([ac.finish_time for ac in finished_aircraft]) / len(finished_aircraft)
        print(f"Mode: {mode}")
        print(f"Finished Aircraft: {len(finished_aircraft)} / {len(aircraft_list)}")
        print(f"Average Flight Time: {avg_time:.2f} s")
    
    if visualize:
        plt.ioff()
        print("Close the visualization window to proceed..." if mode == 'guided' else "Close the visualization window to finish.")
        plt.show() # 保持窗口打开
    
    # Save Data
    if save_data and len(data_buffer) > 0:
        print(f"Saving collected data: {len(data_buffer)} frames to {output_data}")
        # Need adjacency matrix too
        adj = nx.adjacency_matrix(airspace.graph, weight=None).todense()
        # Ensure node ordering matches sorted(nodes) used in collection
        sorted_nodes = sorted(list(airspace.graph.nodes()))
        adj = nx.to_numpy_array(airspace.graph, nodelist=sorted_nodes, weight=None)
        
        np.save(output_data, {
            'features': np.array(data_buffer), # (T, N, F)
            'adj': adj 
        })
        print(f"Data saved to {output_data}")

    return {
        "mode": mode,
        "finished_count": len(finished_aircraft),
        "total_count": len(aircraft_list),
        "avg_time": avg_time
    }

if __name__ == "__main__":
    import sys
    
    # Simple CLI argument parser
    MODE = 'guided'
    if len(sys.argv) > 1:
        MODE = sys.argv[1]

    # 可视化开关
    VISUALIZE = True
    if MODE == 'data_collection':
        VISUALIZE = False
        print("Starting Data Collection Run...")
        
        # Mix of strategies
        # Run 1: Dynamic (60%)
        print("Phase 1: Dynamic Routing Data Collection")
        # We need to tell simulation to use 'dynamic' routing mode for aircraft
        # Passing 'routing_mode' to run_simulation
        metrics1 = run_simulation(mode='data_collection', visualize=False, routing_policy='dynamic')
        
        # We can append data by running multiple times? 
        # Current run_simulation saves directly. We might overwrite.
        # Let's just run one robust session with mixed agents or just Dynamic for now as it's the most valuable.
        # Ideally, we should modify run_simulation to append or accept a policy.
        
    else:
        print("Starting Comparison: Guided vs Free Flight")
        print("------------------------------------------")
        
        print("\nRunning Guided Mode...")
        metrics_guided = run_simulation(mode='guided', visualize=VISUALIZE)
        
        # print("\nRunning Free Flight Mode...")
        # metrics_free = run_simulation(mode='free', visualize=VISUALIZE)
        
        print("\n=== Comparison Results ===")
        print("-" * 56)
        # print(f"{'Finished Aircraft':<20} | {metrics_guided['finished_count']}/{metrics_guided['total_count']:<13} | {metrics_free['finished_count']}/{metrics_free['total_count']:<13}")
        # print(f"{'Avg Flight Time':<20} | {metrics_guided['avg_time']:.2f} s{'':<9} | {metrics_free['avg_time']:.2f} s{'':<9}")
        pass # Placeholder
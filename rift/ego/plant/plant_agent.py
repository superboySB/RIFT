#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : PlanT_agent.py
@Date    : 2023/10/22
"""
import time
import cv2
from pathlib import Path
import warnings
from PIL import Image, ImageDraw, ImageOps
import torch
import math
import shutil

from rift.ego.utils.coordinate_utils import inverse_conversion_2d
from rift.ego.utils.explainability_utils import *
from rift.util.torch_util import CUDA

from rift.ego.plant.data_agent_boxes import DataAgent
from rift.ego.plant.dataset import generate_batch, split_large_BB
from rift.ego.plant.lit_module import LitHFLM

from rift.ego.expert.nav_planner import RoutePlanner_new as RoutePlanner


warnings.filterwarnings("ignore", category=DeprecationWarning)


class PlanTAgent(DataAgent):
    def __init__(self, config, logger):
        super().__init__(config=config)
        self.config = config
        self.logger = logger
        self.exec_or_inter = config['exec_or_inter']
        self.viz_attn_map = config['viz_attn_map']

        # --- 新增：debug文件夹管理 ---
        from pathlib import Path
        import os
        workspace_root = Path(os.environ.get('WORKSPACE_ROOT', '.'))
        self.debug_dir = workspace_root / 'debug'
        if self.debug_dir.exists():
            shutil.rmtree(self.debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.save_mask = []
        self.save_topdowns = []
        self.timings_run_step = []
        self.timings_forward_model = []

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        # 新增：初始化专用log文件
        self.ego_log_path = self.debug_dir / 'collect_pid_data.log'
        if self.ego_log_path.exists():
            self.ego_log_path.unlink()  # 每次运行前清空
        self.ego_log_file = open(self.ego_log_path, 'a')

        # 新增：初始化PID debug csv文件
        self.pid_csv_path = self.debug_dir / 'pid_debug.csv'
        with open(self.pid_csv_path, 'w') as f:
            f.write('step,wp0_x,wp0_y,wp1_x,wp1_y,ego_speed,steer,throttle,brake\n')

        # 新增：用于可视化的PID数据收集
        self.pid_log = {
            'step': [],
            'speed_error': [],
            'steer_error': [],
            'throttle': [],
            'steer': [],
            'brake': [],
            'desired_speed': [],
            'actual_speed': [],
            'waypoint_error': [],
        }
        # 新增：episode级别输入/输出/轨迹收集
        self.episode_inputs = []  # 每步输入状态（自车、它车、路线）
        self.episode_waypoints = []  # 每步planT输出的waypoints
        self.episode_ego_pos = []  # 每步自车实际gps

        # exec_or_inter is used for the interpretability metric
        # exec is the model that executes the actions in carla
        # inter is the model that obtains attention scores and a ranking of the vehicles importance
        LOAD_CKPT_PATH = None
        if self.exec_or_inter is not None:
            if self.exec_or_inter == 'exec':
                LOAD_CKPT_PATH = self.config['exec_model_ckpt_load_path']
        else:
            LOAD_CKPT_PATH = self.config['model_ckpt_load_path']

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            self.net = CUDA(LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH, strict=True, cfg=self.config))
        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.eval()

        # 新增：PlanT输入输出csv收集
        self.io_csv_path = self.debug_dir / 'plant_io_debug.csv'
        self.max_cars = 20
        self.max_routes = 3
        self.num_wps = 4
        header = ['step',
                  'ego_x', 'ego_y', 'ego_yaw', 'ego_speed', 'ego_extent_x', 'ego_extent_y', 'ego_extent_z', 'ego_id']
        header += [f'car{i}_{k}' for i in range(self.max_cars) for k in ['x','y','yaw','speed','extent_x','extent_y','id']]
        header += [f'route{i}_{k}' for i in range(self.max_routes) for k in ['x','y','yaw','id','extent_x','extent_y']]
        header += ['target_point_x', 'target_point_y', 'light_hazard']
        header += [f'wp{i}_x' for i in range(self.num_wps)] + [f'wp{i}_y' for i in range(self.num_wps)]
        # 新增：相对ego坐标系下的特征列
        header += [f'car{i}_rel_x' for i in range(self.max_cars)]
        header += [f'car{i}_rel_y' for i in range(self.max_cars)]
        header += [f'car{i}_rel_yaw' for i in range(self.max_cars)]
        header += [f'car{i}_rel_speed' for i in range(self.max_cars)]
        header += [f'car{i}_rel_extent_x' for i in range(self.max_cars)]
        header += [f'car{i}_rel_extent_y' for i in range(self.max_cars)]
        header += [f'car{i}_rel_id' for i in range(self.max_cars)]
        header += [f'route{i}_rel_x' for i in range(self.max_routes)]
        header += [f'route{i}_rel_y' for i in range(self.max_routes)]
        header += [f'route{i}_rel_yaw' for i in range(self.max_routes)]
        header += [f'route{i}_rel_id' for i in range(self.max_routes)]
        header += [f'route{i}_rel_extent_x' for i in range(self.max_routes)]
        header += [f'route{i}_rel_extent_y' for i in range(self.max_routes)]
        with open(self.io_csv_path, 'w') as f:
            f.write(','.join(header) + '\n')

    def set_planner(self, ego_vehicle, global_plan_gps, global_plan_world_coord):
        super().set_planner(ego_vehicle, global_plan_gps, global_plan_world_coord)
        self._route_planner = RoutePlanner(7.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)

    def tick(self, input_data):
        result = super().tick(input_data)

        waypoint_route = self._route_planner.run_step(result['gps'])

        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]

        ego_target_point = inverse_conversion_2d(target_point, result['gps'], result['compass'])
        result['target_point'] = tuple(ego_target_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, viz_route=None):
        print(f"[PlanTAgent] Step {getattr(self, 'step', 0)} 开始...")
        self.step += 1

        # needed for traffic_light_hazard
        _ = super()._get_brake(stop_sign_hazard=0, vehicle_hazard=0, walker_hazard=0)
        tick_data = self.tick(input_data)
        label_raw = super().get_bev_boxes(input_data=input_data, pos=input_data['gps'], viz_route=viz_route)
        self.episode_inputs.append({
            'label_raw': label_raw,
            'input_data': dict(input_data),
        })
        if 'gps' in input_data:
            self.episode_ego_pos.append(tuple(input_data['gps']))
        else:
            self.episode_ego_pos.append((None, None))

        if self.exec_or_inter == 'exec' or self.exec_or_inter is None:
            self.control = self._get_control(label_raw, tick_data)
        inital_frames_delay = 2
        if self.step < inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # --- 新增：每步写入PID debug csv ---
        if hasattr(self, 'episode_waypoints') and len(self.episode_waypoints) > 0:
            wps = self.episode_waypoints[-1]
            if wps.shape[0] >= 2:
                wp0_x, wp0_y = wps[0][0], wps[0][1]
                wp1_x, wp1_y = wps[1][0], wps[1][1]
            else:
                wp0_x = wp0_y = wp1_x = wp1_y = float('nan')
            ego_speed = input_data.get('speed', float('nan'))
            steer = self.control.steer
            throttle = self.control.throttle
            brake = self.control.brake
            with open(self.pid_csv_path, 'a') as f:
                f.write(f"{self.step},{wp0_x:.4f},{wp0_y:.4f},{wp1_x:.4f},{wp1_y:.4f},{ego_speed:.4f},{steer:.4f},{throttle:.4f},{brake:.4f}\n")

        # --- 优化主log输出 ---
        log_str = f"Step: {self.step}\n"
        # 1. Ego
        ego = label_raw[0]
        log_str += f"Ego State: x={ego['position'][0]:.3f}, y={ego['position'][1]:.3f}, yaw={ego['yaw']:.3f}, speed={ego['speed']:.3f}, extent={ego['extent']}, id={ego['id']}\n"
        # 2. Other vehicles
        others = [v for v in label_raw[1:] if v['class']=='Car']
        log_str += f"Other Vehicles ({len(others)}):\n"
        for v in others:
            log_str += f"  [id={v['id']}] x={v['position'][0]:.3f}, y={v['position'][1]:.3f}, yaw={v['yaw']:.3f}, speed={v['speed']:.3f}, extent={v['extent']}\n"
        # 3. Route points
        routes = [v for v in label_raw if v['class']=='Route']
        log_str += f"Route Points ({len(routes)}):\n"
        for v in routes:
            log_str += f"  [id={v.get('id','?')}] x={v['position'][0]:.3f}, y={v['position'][1]:.3f}, yaw={v['yaw']:.3f}, extent={v['extent']}\n"
        # 4. PlanT输出waypoints
        if hasattr(self, 'episode_waypoints') and len(self.episode_waypoints) > 0:
            wps = self.episode_waypoints[-1]
            log_str += f"Waypoints (future, local frame):\n"
            for idx, wp in enumerate(wps):
                log_str += f"  [{idx}]: x={wp[0]:.3f}, y={wp[1]:.3f}\n"
        # 5. PID控制量
        log_str += f"Control: steer={self.control.steer:.4f}, throttle={self.control.throttle:.4f}, brake={self.control.brake:.4f}\n"
        log_str += f"Ego speed: {input_data.get('speed', None):.3f}\n"
        log_str += f"Ego GPS: {input_data.get('gps', None)}\n"
        log_str += "-"*40 + "\n"
        self.ego_log_file.write(log_str)
        self.ego_log_file.flush()
        print(f"[PlanTAgent] Step {self.step} 输入状态和控制量已写入log。")

        # --- 优化可视化 ---
        N = 20
        if self.step % N == 0:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                fig, ax = plt.subplots(figsize=(10, 10))
                # 画ego bbox（红色）
                ego = label_raw[0]
                def plot_bbox(ax, x, y, yaw, extent_x, extent_y, color, label=None, alpha=1.0, lw=2, zorder=2):
                    # extent_x/extent_y为bbox长宽（半长半宽）
                    from matplotlib.patches import Polygon
                    c, s = np.cos(yaw), np.sin(yaw)
                    dx = extent_x
                    dy = extent_y
                    corners = np.array([
                        [dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]
                    ])
                    rot = np.array([[c, -s], [s, c]])
                    corners = corners @ rot.T + np.array([x, y])
                    poly = Polygon(corners, closed=True, edgecolor=color, facecolor='none', lw=lw, alpha=alpha, zorder=zorder, label=label)
                    ax.add_patch(poly)
                # ego bbox
                plot_bbox(ax, 0, 0, ego['yaw'], ego['extent'][2]/2, ego['extent'][1]/2, color='r', label='Ego', lw=2, zorder=3)
                # 画其他车辆bbox（绿色）
                for idx, v in enumerate(others):
                    rel_x, rel_y = v['position'][0], v['position'][1]
                    plot_bbox(ax, rel_x, rel_y, v['yaw'], v['extent'][2]/2, v['extent'][1]/2, color='g', label='Other Car' if idx==0 else None, lw=1.5, zorder=2)
                # 画route点bbox（蓝色）
                for idx, v in enumerate(routes):
                    rel_x, rel_y = v['position'][0], v['position'][1]
                    plot_bbox(ax, rel_x, rel_y, v['yaw'], v['extent'][2]/2, v['extent'][1]/2, color='b', label='Route Pt' if idx==0 else None, lw=1.5, zorder=1)
                # 画waypoints（蓝色点连线）
                if hasattr(self, 'episode_waypoints') and len(self.episode_waypoints) > 0:
                    wps = np.array(self.episode_waypoints[-1])
                    ax.plot(wps[:,0], wps[:,1], 'bo-', label='Pred Waypoints', zorder=4)
                    ax.scatter(wps[:,0], wps[:,1], c='b', zorder=5)
                # 速度、step等信息
                ax.set_title(f"Step {self.step} | Speed: {input_data.get('speed', 0):.2f} m/s")
                ax.set_xlabel('x (local)')
                ax.set_ylabel('y (local)')
                ax.axis('equal')
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                ax.grid(True)
                plt.tight_layout()
                fig.savefig(self.debug_dir / f'step_{self.step:04d}_viz.png')
                plt.close(fig)
            except Exception as e:
                print(f"[PlanTAgent] Failed to plot debug figure at step {self.step}: {e}")

        print(f"[PlanTAgent] Step {self.step} 结束，控制量: steer={self.control.steer:.4f}, throttle={self.control.throttle:.4f}, brake={self.control.brake:.4f}")
        return self.control

    def _get_control(self, label_raw, input_data):
        
        gt_velocity = CUDA(torch.FloatTensor([input_data['speed']]))

        # input_data contains [speed, imu(yaw angle), gps(x, y location)]
        x, y, _, tp, light = self.get_input_batch(label_raw, input_data)
    
        _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)

        steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # 新增：保存waypoints
        self.episode_waypoints.append(pred_wp[0].cpu().numpy())

        # 新增：PlanT输入输出csv收集
        import math
        row = [getattr(self, 'step', -1)]
        # ego
        ego = label_raw[0]
        row += [ego['position'][0], ego['position'][1], ego['yaw'], ego['speed'],
                ego['extent'][0], ego['extent'][1], ego['extent'][2], ego['id']]
        # cars
        cars = [v for v in label_raw[1:] if v['class']=='Car']
        for i in range(self.max_cars):
            if i < len(cars):
                v = cars[i]
                row += [v['position'][0], v['position'][1], v['yaw'], v['speed'],
                        v['extent'][0], v['extent'][1], v['id']]
            else:
                row += [math.nan]*7
        # routes (route point指的是class为'Route'的元素)
        routes = [v for v in label_raw if v['class']=='Route']
        for i in range(self.max_routes):
            if i < len(routes):
                v = routes[i]
                row += [v['position'][0], v['position'][1], v['yaw'], v['id'],
                        v['extent'][0], v['extent'][1]]
            else:
                row += [math.nan]*6
        # target point
        row += [input_data['target_point'][0], input_data['target_point'][1]]
        # light
        row += [self.traffic_light_hazard]
        # waypoints
        wps = pred_wp[0].cpu().numpy()
        for i in range(self.num_wps):
            if i < wps.shape[0]:
                row += [wps[i][0], wps[i][1]]
            else:
                row += [math.nan, math.nan]
        # 新增：相对ego坐标系下的特征
        import numpy as np
        def global_to_rel(x, y, ego_x, ego_y, ego_yaw):
            dx = x - ego_x
            dy = y - ego_y
            # ego_yaw为弧度
            rel_x = dx * np.cos(-ego_yaw) - dy * np.sin(-ego_yaw)
            rel_y = dx * np.sin(-ego_yaw) + dy * np.cos(-ego_yaw)
            return rel_x, rel_y
        ego_x, ego_y, ego_yaw = ego['position'][0], ego['position'][1], ego['yaw']
        # cars
        for i in range(self.max_cars):
            if i < len(cars):
                v = cars[i]
                rel_x, rel_y = global_to_rel(v['position'][0], v['position'][1], ego_x, ego_y, ego_yaw)
                rel_yaw = v['yaw'] - ego_yaw
                rel_speed = v['speed'] # 保持原速，若需相对速度可改
                rel_extent_x = v['extent'][0]
                rel_extent_y = v['extent'][1]
                rel_id = v['id']
                row += [rel_x, rel_y, rel_yaw, rel_speed, rel_extent_x, rel_extent_y, rel_id]
            else:
                row += [math.nan]*7
        # routes
        for i in range(self.max_routes):
            if i < len(routes):
                v = routes[i]
                rel_x, rel_y = global_to_rel(v['position'][0], v['position'][1], ego_x, ego_y, ego_yaw)
                rel_yaw = v['yaw'] - ego_yaw
                rel_id = v['id']
                rel_extent_x = v['extent'][0]
                rel_extent_y = v['extent'][1]
                row += [rel_x, rel_y, rel_yaw, rel_id, rel_extent_x, rel_extent_y]
            else:
                row += [math.nan]*6
        with open(self.io_csv_path, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')

        # 只保留一次详细log，去掉重复log输出
        log_str = f"Step: {getattr(self, 'step', -1)}\n"
        ego = label_raw[0]
        log_str += f"Ego State: x={ego['position'][0]:.3f}, y={ego['position'][1]:.3f}, yaw={ego['yaw']:.3f}, speed={ego['speed']:.3f}, extent={ego['extent']}, id={ego['id']}\n"
        others = [v for v in label_raw[1:] if v['class']=='Car']
        log_str += f"Other Vehicles ({len(others)}):\n"
        for v in others:
            log_str += f"  [id={v['id']}] x={v['position'][0]:.3f}, y={v['position'][1]:.3f}, yaw={v['yaw']:.3f}, speed={v['speed']:.3f}, extent={v['extent']}\n"
        routes = [v for v in label_raw if v['class']=='Route']
        log_str += f"Route Points ({len(routes)}):\n"
        for v in routes:
            log_str += f"  [id={v.get('id','?')}] x={v['position'][0]:.3f}, y={v['position'][1]:.3f}, yaw={v['yaw']:.3f}, extent={v['extent']}\n"
        log_str += f"Waypoints (future, local frame):\n"
        for idx, wp in enumerate(pred_wp[0].cpu().numpy()):
            log_str += f"  [{idx}]: x={wp[0]:.3f}, y={wp[1]:.3f}\n"
        log_str += f"Control: steer={control.steer:.4f}, throttle={control.throttle:.4f}, brake={control.brake:.4f}\n"
        log_str += "-"*40 + "\n"
        self.ego_log_file.write(log_str)
        self.ego_log_file.flush()
        print(f"[PlanTAgent] Step {getattr(self, 'step', -1)} 输入状态和控制量已写入log。")

        # 新增：记录waypoints和控制量到log文件
        log_str = f"Step: {getattr(self, 'step', -1)}\n"
        log_str += f"Waypoints (future, local frame):\n"
        for idx, wp in enumerate(pred_wp[0].cpu().numpy()):
            log_str += f"  [{idx}]: x={wp[0]:.3f}, y={wp[1]:.3f}\n"
        log_str += f"Control: steer={control.steer:.4f}, throttle={control.throttle:.4f}, brake={control.brake:.4f}\n"
        log_str += "-"*40 + "\n"
        self.ego_log_file.write(log_str)
        self.ego_log_file.flush()

        # 新增：收集PID相关数据用于可视化
        # 速度误差、目标速度、实际速度
        speed = float(input_data['speed'])
        wp_np = pred_wp[0].cpu().numpy()
        if wp_np.shape[0] > 1:
            desired_speed = float(np.linalg.norm(wp_np[1] - wp_np[0]))
        else:
            desired_speed = 0.0
        speed_error = desired_speed - speed
        # 轨迹跟踪误差（自车当前位置到第一个waypoint的距离）
        if 'gps' in input_data:
            ego_pos = np.array(input_data['gps'][:2])
            wp0 = wp_np[0]
            waypoint_error = float(np.linalg.norm(wp0 - ego_pos))
        else:
            waypoint_error = 0.0
        # 转向误差（用angle近似）
        aim = (wp_np[1] + wp_np[0]) / 2.0 if wp_np.shape[0] > 1 else wp_np[0]
        steer_error = float(np.degrees(np.arctan2(aim[1], aim[0])) / 90)
        self.pid_log['step'].append(getattr(self, 'step', -1))
        self.pid_log['speed_error'].append(speed_error)
        self.pid_log['steer_error'].append(steer_error)
        self.pid_log['throttle'].append(float(throttle))
        self.pid_log['steer'].append(float(steer))
        self.pid_log['brake'].append(float(brake))
        self.pid_log['desired_speed'].append(desired_speed)
        self.pid_log['actual_speed'].append(speed)
        self.pid_log['waypoint_error'].append(waypoint_error)

        viz_trigger = ((self.step % 20 == 0) and self.cfg['viz'])
        if viz_trigger and self.step > 2:
            create_BEV(label_raw, light, tp, pred_wp)

        if self.viz_attn_map:
            attn_vector = get_attn_norm_vehicles(self.cfg['attention_score'], self.data_car, attn_map)
            keep_vehicle_ids, attn_indices, keep_vehicle_attn = get_vehicleID_from_attn_scores(self.data, self.data_car, self.cfg['topk'], attn_vector)
            draw_attention_bb_in_carla(self._world, keep_vehicle_ids, keep_vehicle_attn)

        return control
    
    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        if self.config['training']['input_ego']:
            data = label_raw
        else:
            data = label_raw[1:] # remove first element (ego vehicle)

        data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data if x['class'] == 'Car']
        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2., # type indicator for route
                float(x['position'][0])-float(label_raw[0]['position'][0]),
                float(x['position'][1])-float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ] 
            for j, x in enumerate(data)
            if x['class'] == 'Route' 
            and float(x['id']) < self.config['training']['max_NextRouteBBs']]
        
        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:self.config['training']['max_NextRouteBBs']]

        assert len(data_route) <= self.config['training']['max_NextRouteBBs'], 'Too many routes'

        if self.config['training']['remove_velocity'] == 'input':
            for i in range(len(data_car)):
                data_car[i][4] = 0.

        if self.config['training']['route_only_wp']:
            for i in range(len(data_route)):
                data_route[i][3] = 0.
                data_route[i][-2] = 0.
                data_route[i][-1] = 0.

        features = data_car + data_route

        sample['input'] = features

        # dummy data
        sample['output'] = features
        sample['light'] = self.traffic_light_hazard

        local_command_point = np.array([input_data['target_point'][0], input_data['target_point'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]
        
        input_batch = generate_batch(batch)
        
        self.data = data
        self.data_car = data_car
        self.data_route = data_route
        
        return input_batch

    def destroy(self):
        super().destroy()
        del self.net
        # --- 新增：关闭log文件 ---
        if hasattr(self, 'ego_log_file'):
            self.ego_log_file.close()
        # 不再在destroy时画大图
        # 新增：绘制PID相关曲线
        try:
            import matplotlib.pyplot as plt
            import os
            from pathlib import Path
            workspace_root = Path(os.environ.get('WORKSPACE_ROOT', '.'))
            fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
            axs[0].plot(self.pid_log['step'], self.pid_log['desired_speed'], label='Desired Speed')
            axs[0].plot(self.pid_log['step'], self.pid_log['actual_speed'], label='Actual Speed')
            axs[0].set_ylabel('Speed (m/s)')
            axs[0].set_title('Speed Tracking')
            axs[0].legend()
            axs[1].plot(self.pid_log['step'], self.pid_log['speed_error'], label='Speed Error')
            axs[1].set_ylabel('Speed Error (m/s)')
            axs[1].set_title('Speed Error')
            axs[2].plot(self.pid_log['step'], self.pid_log['steer_error'], label='Steer Error (norm)')
            axs[2].plot(self.pid_log['step'], self.pid_log['steer'], label='Steer Output')
            axs[2].set_ylabel('Steer')
            axs[2].set_title('Steer Tracking')
            axs[2].legend()
            axs[3].plot(self.pid_log['step'], self.pid_log['waypoint_error'], label='Waypoint Error')
            axs[3].set_ylabel('Waypoint Error (m)')
            axs[3].set_xlabel('Step')
            axs[3].set_title('Waypoint Tracking Error')
            axs[3].legend()
            plt.tight_layout()
            fig.savefig(workspace_root / 'collect_pid_data_plot.png')
            plt.close(fig)
            print(f"[PlanTAgent] PID相关曲线已保存到 {workspace_root / 'collect_pid_data_plot.png'}")
        except Exception as e:
            print(f"[PlanTAgent] Failed to plot PID curves: {e}")
        # 新增：episode级别轨迹可视化
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            workspace_root = Path(os.environ.get('WORKSPACE_ROOT', '.') )
            fig, ax = plt.subplots(figsize=(10, 10))
            # 画所有step的waypoints轨迹
            for i, wps in enumerate(self.episode_waypoints):
                wps = np.array(wps)
                ax.plot(wps[:,0], wps[:,1], color='blue', alpha=0.2, linewidth=1)
            # 画自车实际轨迹
            ego_pos = np.array([p for p in self.episode_ego_pos if p[0] is not None])
            if len(ego_pos) > 0:
                ax.plot(ego_pos[:,0], ego_pos[:,1], color='red', marker='o', label='Ego Actual Trajectory')
            ax.set_title('PlanT Predicted Waypoints (blue, all steps) vs Ego Actual Trajectory (red)')
            ax.set_xlabel('x (local)')
            ax.set_ylabel('y (local)')
            ax.legend()
            plt.tight_layout()
            fig.savefig(workspace_root / 'collect_pid_episode_traj.png')
            plt.close(fig)
            print(f"[PlanTAgent] Episode轨迹可视化已保存到 {workspace_root / 'collect_pid_episode_traj.png'}")
        except Exception as e:
            print(f"[PlanTAgent] Failed to plot episode trajectory: {e}")


def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, pix_per_m=5):

    pred_wp = np.array(pred_wp.squeeze())
    s=0
    max_d = 30
    size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
    PIXELS_PER_METER = pix_per_m

    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]

    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]
    
    for ix, sequence in enumerate([labels_org]):
               
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            # if vehicle['class'] != 'Car':
            #     continue
            
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                    ix = 0
                else:                
                    draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                imgs[ix] = Image.fromarray(image)
                
    for wp in pred_wp:
        x = wp[1]*PIXELS_PER_METER + origin[1]
        y = -wp[0]*PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)
          
    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1]*PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0])*PIXELS_PER_METER + origin[0]  
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)
    
    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)
    
    ## add rgb image and lidar
    # all_images = np.concatenate((images_lidar, np.array(img_final)), axis=1)
    # all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = img_final
    
    Path(f'bev_viz').mkdir(parents=True, exist_ok=True)
    all_images.save(f'bev_viz/{time.time()}_{s}.png')

    # return BEV


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2  


def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : extract_complete_npz_data.py
@Author  : AI Assistant
@Date    : 2024/12/19
@Desc    : 提取NPZ文件的原始数据结构，生成JSON数据文件和TXT解释文件
"""

import numpy as np
import json
import os
from shapely.geometry import Polygon


def extract_npz_structure(town_name="Town01"):
    """提取NPZ文件的原始数据结构"""
    
    filename = f"data/map_data/{town_name}_HD_map.npz"
    
    print(f"正在提取 {filename} 的原始数据结构...")
    
    # 加载数据
    data = np.load(filename, allow_pickle=True)
    lane_marking_dict = dict(data['arr'])
    
    # 原始数据结构
    raw_data = {
        'file_info': {
            'town_name': town_name,
            'file_size_mb': round(os.path.getsize(filename) / (1024*1024), 2),
            'total_top_level_keys': len(lane_marking_dict),
            'all_top_level_keys': list(lane_marking_dict.keys())
        },
        'roads': {},
        'crosswalks': []
    }
    
    road_count = 0
    
    # 逐个提取道路数据
    for road_id, road_data in lane_marking_dict.items():
        if road_id == 'Crosswalks':
            # 处理人行横道
            crosswalks = road_data
            for i, crosswalk in enumerate(crosswalks):
                crosswalk_data = {}
                
                # 提取所有字段
                for key, value in crosswalk.items():
                    if key == 'Polygon' and hasattr(value, 'exterior'):
                        # 处理多边形数据
                        coords = list(value.exterior.coords)
                        crosswalk_data[key] = {
                            'type': 'Polygon',
                            'coordinates': coords,
                            'area': value.area,
                            'bounds': value.bounds
                        }
                    elif key == 'Location':
                        crosswalk_data[key] = list(value) if hasattr(value, '__iter__') else value
                    else:
                        crosswalk_data[key] = value
                
                raw_data['crosswalks'].append(crosswalk_data)
            continue
        
        # 处理道路数据
        road_count += 1
        
        # 只详细展示前5个道路
        if road_count <= 5:
            road_structure = {
                'road_id': road_id,
                'lanes': {},
                'trigger_volumes': []
            }
            
            for lane_id, lane_data in road_data.items():
                if lane_id == 'Trigger_Volumes':
                    # 处理触发区域
                    trigger_volumes = lane_data
                    for tv in trigger_volumes:
                        tv_data = {}
                        for key, value in tv.items():
                            if key == 'Polygon' and hasattr(value, 'exterior'):
                                coords = list(value.exterior.coords)
                                tv_data[key] = {
                                    'type': 'Polygon',
                                    'coordinates': coords,
                                    'area': value.area,
                                    'bounds': value.bounds
                                }
                            elif key == 'ParentActor_Location':
                                tv_data[key] = list(value) if hasattr(value, '__iter__') else value
                            elif key == 'Points':
                                tv_data[key] = list(value) if hasattr(value, '__iter__') else value
                            else:
                                tv_data[key] = value
                        road_structure['trigger_volumes'].append(tv_data)
                    continue
                
                # 处理车道数据
                lane_structure = {
                    'lane_id': lane_id,
                    'lane_type': lane_data.get('LaneType'),
                    'lane_width': lane_data.get('LaneWidth'),
                    'lane_markings': {}
                }
                
                # 处理车道标记
                lane_marks = lane_data.get('LaneMark', {})
                for position, marks in lane_marks.items():  # Center, Left, Right
                    lane_structure['lane_markings'][position] = []
                    
                    for mark in marks:
                        mark_data = {
                            'type': mark.get('Type'),
                            'color': mark.get('Color'),
                            'topology_type': mark.get('TopologyType'),
                            'topology_connections': mark.get('Topology', []),
                            'left_adjacent_lane': mark.get('Left'),
                            'right_adjacent_lane': mark.get('Right'),
                            'waypoints': []
                        }
                        
                        # 处理waypoint数据 - 只取前10个作为示例
                        points = mark.get('Points', [])
                        for i, point in enumerate(points[:10]):
                            if len(point) >= 3:
                                waypoint = {
                                    'index': i,
                                    'location': {
                                        'x': float(point[0][0]),
                                        'y': float(point[0][1]),
                                        'z': float(point[0][2])
                                    },
                                    'rotation': {
                                        'roll': float(point[1][0]),
                                        'pitch': float(point[1][1]),
                                        'yaw': float(point[1][2])
                                    },
                                    'is_junction': bool(point[2])
                                }
                                mark_data['waypoints'].append(waypoint)
                        
                        # 添加waypoint总数信息
                        mark_data['total_waypoints'] = len(points)
                        
                        lane_structure['lane_markings'][position].append(mark_data)
                
                road_structure['lanes'][str(lane_id)] = lane_structure
            
            raw_data['roads'][str(road_id)] = road_structure
    
    # 添加统计信息
    raw_data['statistics'] = {
        'total_roads': road_count,
        'total_crosswalks': len(raw_data['crosswalks']),
        'roads_shown_in_detail': min(5, road_count),
        'note': 'Only first 5 roads shown in detail, and only first 10 waypoints per lane marking'
    }
    
    return raw_data


def generate_explanation_file(town_name="Town01"):
    """生成数据结构解释文件"""
    
    explanation = f"""
HD Map 数据结构解释文件
=====================================
城镇: {town_name}
生成时间: {os.path.basename(__file__)} 运行结果

1. 文件基本信息 (file_info)
   - town_name: 城镇名称
   - file_size_mb: 文件大小（MB）
   - total_top_level_keys: 顶层数据项总数
   - all_top_level_keys: 所有顶层键值列表

2. 道路数据结构 (roads)
   每个道路ID包含：
   - road_id: CARLA道路ID
   - lanes: 车道信息字典
     - lane_id: 车道ID（负数表示相对于道路中心线的位置）
     - lane_type: 车道类型（通常为 'Driving'）
     - lane_width: 车道宽度（米）
     - lane_markings: 车道标记信息
       - Center: 中心标记
       - Left: 左侧标记  
       - Right: 右侧标记
       
   车道标记详细信息：
   - type: 标记类型
     * 'Center': 中心标记
     * 'Broken': 虚线
     * 'Solid': 实线
     * 'SolidSolid': 双实线
     * 'NONE': 无标记
   - color: 标记颜色
     * 'White': 白色
     * 'Yellow': 黄色
     * 'Blue': 蓝色
   - topology_type: 拓扑类型（仅中心标记有此字段）
     * 'Normal': 正常路段
     * 'Junction': 交叉口
     * 'EnterNormal': 进入正常路段
     * 'EnterJunction': 进入交叉口
   - topology_connections: 拓扑连接信息，格式为 [[road_id, lane_id], ...]
   - left_adjacent_lane: 左侧相邻车道 [road_id, lane_id]
   - right_adjacent_lane: 右侧相邻车道 [road_id, lane_id]
   - waypoints: 路径点列表
     * location: 位置坐标 (x, y, z)，单位米
     * rotation: 旋转角度 (roll, pitch, yaw)，单位弧度
     * is_junction: 是否在交叉口
   - total_waypoints: 总waypoint数量

   - trigger_volumes: 触发区域列表（交通控制设施）
     - type: 父对象类型
       * 'TrafficLight': 交通信号灯
       * 'StopSign': 停车标志
     - ParentActor_Location: 父对象位置 [x, y, z]
     - Polygon: 触发区域多边形
       * coordinates: 多边形顶点坐标列表
       * area: 面积（平方米）
       * bounds: 边界框 (min_x, min_y, max_x, max_y)

3. 人行横道数据 (crosswalks)
   每个人行横道包含：
   - Location: 中心位置 [x, y]
   - Polygon: 人行横道多边形区域
     * coordinates: 多边形顶点坐标列表
     * area: 面积（平方米）
     * bounds: 边界框

4. 统计信息 (statistics)
   - total_roads: 总道路数
   - total_crosswalks: 总人行横道数
   - roads_shown_in_detail: 详细展示的道路数（限制为前5个）
   - note: 数据展示说明

5. 坐标系统
   - 使用右手坐标系
   - X轴：东向为正
   - Y轴：北向为正
   - Z轴：上向为正
   - 角度单位：弧度
   - 距离单位：米

6. 车道ID说明
   - 负数ID：表示与道路方向相反的车道
   - 正数ID：表示与道路方向相同的车道
   - 绝对值越大，离道路中心线越远

7. 数据使用注意事项
   - 本JSON仅展示前5个道路的详细信息
   - 每个车道标记仅展示前10个waypoint
   - 完整数据包含所有道路和所有waypoint
   - 人行横道和触发区域数据完整展示
"""
    
    return explanation


def main():
    """主函数"""
    
    # town_name = "Town01"
    town_name = "Town03"
    
    # 提取数据结构
    raw_data = extract_npz_structure(town_name)
    
    # 保存JSON数据文件
    json_file = f"{town_name}_hd_map_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    
    # 生成并保存解释文件
    explanation = generate_explanation_file(town_name)
    txt_file = f"hd_map_explanation.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(explanation)
    
    # 打印简要信息
    print(f"数据文件已保存: {json_file}")
    print(f"解释文件已保存: {txt_file}")
    print(f"\n基本信息:")
    print(f"  城镇: {raw_data['file_info']['town_name']}")
    print(f"  文件大小: {raw_data['file_info']['file_size_mb']} MB")
    print(f"  总道路数: {raw_data['statistics']['total_roads']}")
    print(f"  人行横道数: {raw_data['statistics']['total_crosswalks']}")
    print(f"  详细展示道路: {raw_data['statistics']['roads_shown_in_detail']}")
    
    # 显示车道统计
    total_lanes = 0
    for road_id, road_data in raw_data['roads'].items():
        total_lanes += len(road_data['lanes'])
    
    print(f"  前5个道路的车道总数: {total_lanes}")
    
    # 显示第一个道路的结构示例
    if raw_data['roads']:
        first_road_id = list(raw_data['roads'].keys())[0]
        first_road = raw_data['roads'][first_road_id]
        print(f"\n第一个道路示例 (ID: {first_road_id}):")
        print(f"  车道数: {len(first_road['lanes'])}")
        print(f"  触发区域数: {len(first_road['trigger_volumes'])}")
        
        if first_road['lanes']:
            first_lane_id = list(first_road['lanes'].keys())[0]
            first_lane = first_road['lanes'][first_lane_id]
            print(f"  第一个车道 (ID: {first_lane_id}):")
            print(f"    类型: {first_lane['lane_type']}")
            print(f"    宽度: {first_lane['lane_width']}m")
            print(f"    标记数: {len(first_lane['lane_markings'])}")


if __name__ == "__main__":
    main()
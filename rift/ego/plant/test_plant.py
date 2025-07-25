'''
PlanT 离线推理/对比测试模块
============================

功能：
    - 读取 data/plant_debug.csv（包含每步的PlanT输入和输出）
    - 用csv中的输入拼成PlanT模型输入，调用PlanT模型（加载权重），输出预测的waypoints
    - 将预测waypoints与csv中的真实waypoints逐步对比，输出每步的预测值、真值和误差
    - 统计所有步的平均绝对误差，便于验证PlanT模型的离线复现正确性
    - 只打印误差大于阈值（THRESHOLD）的行，便于聚焦异常
    - 支持将指定step区间的waypoint对比过程保存为mp4动画，直观展示轨迹预测与真实对比

用法示例：
    # 运行方式（需在RIFT根目录下）
    python rift/ego/plant/test_plant.py --csv data/plant_debug.csv --ckpt rift/ego/model_ckpt/PlanT_medium/checkpoints/PlanT_pretrain.ckpt --save_mp4 out_plant.mp4 --start 0 --end 5

依赖说明：
    - 需已安装matplotlib、torch、ffmpeg（系统需有ffmpeg命令）。
    - 不再弹窗显示，只保存mp4动画。
'''
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from transformers import AutoConfig, AutoModel
from collections import deque

# ==== 复制HFLM及其依赖，完全本地化推理，无外部依赖 ====
class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net
        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y
        precisions = [
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_angle", 4),
            self.config_all['pre_training'].get("precision_speed", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
        ]
        self.vocab_size = [2**i for i in precisions]
        config = AutoConfig.from_pretrained(self.config_net['hf_checkpoint'])
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)
        self.cls_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))
        self.eos_emb = nn.Parameter(torch.randn(1, self.num_attributes + 1))
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        self.obj_token = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.num_attributes)) for _ in range(self.object_types)
        ])
        self.obj_emb = nn.ModuleList([
            nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)
        ])
        self.drop = nn.Dropout(config_net['embd_pdrop'])
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=65)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(65, 2)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def pad_sequence_batch(self, x_batched):
        # 只处理单batch推理（x_batched: [1, N, 7]）
        assert x_batched.dim() == 3 and x_batched.shape[0] == 1, "只支持单batch推理。"
        x_tokens_batch = x_batched[0]  # [N, 7]
        x_seq = torch.cat([self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)  # [N+2, 7]
        input_batch = x_seq.unsqueeze(0)  # [1, N+2, 7]
        return input_batch
    def forward(self, idx, target=None, target_point=None, light_hazard=None):
        x_batched = torch.cat(idx, dim=0)
        input_batch = self.pad_sequence_batch(x_batched)
        input_batch_type = input_batch[:, :, 0]
        input_batch_data = input_batch[:, :, 1:]
        car_mask = (input_batch_type == 1).unsqueeze(-1)
        route_mask = (input_batch_type == 2).unsqueeze(-1)
        other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not())
        masks = [car_mask, route_mask, other_mask]
        (B, O, A) = (input_batch_data.shape)
        input_batch_data = rearrange(input_batch_data, "b objects attributes -> (b objects) attributes")
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)
        obj_embeddings = [self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)]
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)
        x = self.drop(embedding)
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=False)
        x = output.last_hidden_state
        z = self.wp_head(x[:, 0, :])
        z = torch.cat((z, light_hazard), 1)
        output_wp = list()
        x_wp = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)
        for _ in range(self.config_all['training']['pred_len']):
            x_in = torch.cat([x_wp, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x_wp = dx + x_wp
            output_wp.append(x_wp)
        pred_wp = torch.stack(output_wp, dim=1)
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3
        return None, None, pred_wp, None

def load_hflm_from_checkpoint(ckpt_path, cfg):
    model = HFLM(cfg['network'], cfg)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('==== 权重加载调试 ====')
    print('Missing keys:', missing, '\nUnexpected keys:', unexpected)
    print('==== 权重文件 keys 示例 ====')
    print(list(state_dict.keys())[:20])
    return model

# ==== END ==== 独立推理模型定义

def build_input_from_csv_row(row, max_cars=20, max_routes=3):
    import numpy as np
    import torch
    data_car = []
    for i in range(max_cars):
        rel_x = float(row.get(f'car{i}_rel_x', 0))
        rel_y = float(row.get(f'car{i}_rel_y', 0))
        rel_yaw = float(row.get(f'car{i}_rel_yaw', 0))
        rel_speed = float(row.get(f'car{i}_rel_speed', 0))
        rel_extent_x = float(row.get(f'car{i}_rel_extent_x', 0))
        rel_extent_y = float(row.get(f'car{i}_rel_extent_y', 0))
        # 补0
        rel_x = 0 if np.isnan(rel_x) else rel_x
        rel_y = 0 if np.isnan(rel_y) else rel_y
        rel_yaw = 0 if np.isnan(rel_yaw) else rel_yaw
        rel_speed = 0 if np.isnan(rel_speed) else rel_speed
        rel_extent_x = 0 if np.isnan(rel_extent_x) else rel_extent_x
        rel_extent_y = 0 if np.isnan(rel_extent_y) else rel_extent_y
        rel_speed_kmh = rel_speed * 3.6  # plant_agent输入是km/h
        rel_yaw_deg = rel_yaw * 180 / np.pi  # plant_agent输入是角度
        data_car.append([
            1.0, rel_x, rel_y, rel_yaw_deg, rel_speed_kmh, rel_extent_x, rel_extent_y
        ])
    data_route = []
    for i in range(max_routes):
        rel_x = float(row.get(f'route{i}_rel_x', 0))
        rel_y = float(row.get(f'route{i}_rel_y', 0))
        rel_yaw = float(row.get(f'route{i}_rel_yaw', 0))
        rel_id = float(row.get(f'route{i}_rel_id', 0))
        rel_extent_x = float(row.get(f'route{i}_rel_extent_x', 0))
        rel_extent_y = float(row.get(f'route{i}_rel_extent_y', 0))
        rel_x = 0 if np.isnan(rel_x) else rel_x
        rel_y = 0 if np.isnan(rel_y) else rel_y
        rel_yaw = 0 if np.isnan(rel_yaw) else rel_yaw
        rel_id = 0 if np.isnan(rel_id) else rel_id
        rel_extent_x = 0 if np.isnan(rel_extent_x) else rel_extent_x
        rel_extent_y = 0 if np.isnan(rel_extent_y) else rel_extent_y
        rel_yaw_deg = rel_yaw * 180 / np.pi
        data_route.append([
            2.0, rel_x, rel_y, rel_yaw_deg, rel_id, rel_extent_x, rel_extent_y
        ])
    features = data_car + data_route
    features = np.array(features, dtype=np.float32)
    # 已补齐，无需再pad
    target_point = np.array([
        float(row.get('target_point_x', 0)),
        float(row.get('target_point_y', 0))
    ], dtype=np.float32)
    if np.isnan(target_point).any():
        target_point = np.zeros_like(target_point)
    light_hazard_raw = row.get('light_hazard', 0.0)
    if isinstance(light_hazard_raw, str):
        if light_hazard_raw.lower() == 'false':
            light_hazard = 0.0
        elif light_hazard_raw.lower() == 'true':
            light_hazard = 1.0
        else:
            light_hazard = float(light_hazard_raw)
    else:
        light_hazard = float(light_hazard_raw)
    if np.isnan(light_hazard):
        light_hazard = 0.0
    x = torch.from_numpy(features).unsqueeze(0).float()  # [1, N, 7]
    y = torch.zeros_like(x)  # [1, N, 7]，dummy
    tp = torch.from_numpy(target_point).unsqueeze(0).float()  # [1, 2]
    light = torch.tensor([light_hazard]).unsqueeze(0).float()  # [1, 1]
    return x, y, tp, light

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test PlanT model with CSV input')
    parser.add_argument('--csv', type=str, required=True, help='Path to plant_debug.csv')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to PlanT model .ckpt')
    parser.add_argument('--threshold', type=float, default=0.1, help='Only print rows with any waypoint error above this threshold (meter)')
    parser.add_argument('--save_mp4', type=str, default=None, help='Path to save mp4 animation (all steps, fps=10)')
    parser.add_argument('--start', type=int, default=0, help='Start step for mp4 output (inclusive), default 0')
    parser.add_argument('--end', type=int, default=500, help='End step for mp4 output (exclusive), default 500')
    args = parser.parse_args()

    # ==== mock config ====
    cfg = {
        'network': {
            'hf_checkpoint': "rift/ego/model_ckpt/PlanT_medium/checkpoints/bert-medium",
            'embd_pdrop': 0.1,
        },
        'training': {
            'pred_len': 4,
        },
        'pre_training': {
            'precision_pos': 7,
            'precision_speed': 4,
            'precision_angle': 5,
            'pretraining': 'forecast',
            'multitask': True,
        },
    }

    # 加载模型（只用HFLM，避免循环依赖）
    model = load_hflm_from_checkpoint(args.ckpt, cfg)
    model.eval()
    model.cuda()
    # 检查模型参数是否有nan
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'[参数NAN] Param {name} contains nan!')

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 只处理start:end区间
    rows = rows[args.start:args.end]

    print(f"{'step':>5} | {'wp_errs (m)':<40} | {'max_err':>8}")
    print('-'*60)
    all_errs = []
    pred_wps_list = []
    true_wps_list = []

    for idx, row in enumerate(rows):
        step = row.get('step', idx+args.start)
        x, y, tp, light = build_input_from_csv_row(row)
        print(f"==== Step {row['step']} 输入调试 ====")
        print('x:', x)
        print('tp:', tp)
        print('light:', light)
        print('x.isnan:', torch.isnan(x).any(), 'tp.isnan:', torch.isnan(tp).any(), 'light.isnan:', torch.isnan(light).any())
        x, y, tp, light = x.cuda(), y.cuda(), tp.cuda(), light.cuda()
        with torch.no_grad():
            # forward前后调试
            print('==== Forward前参数检查 ====')
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f'[参数NAN] Param {name} contains nan!')
            _, _, pred_wp, _ = model([x], [y], target_point=tp, light_hazard=light)
            print('==== Forward后输出调试 ====')
            print('pred_wp:', pred_wp)
            if torch.isnan(pred_wp).any():
                print('[输出NAN] pred_wp contains nan!')
        pred_wp = pred_wp[0].cpu().numpy()
        print(f"Step {row['step']} pred_wp: {pred_wp}")
        true_wp = []
        for i in range(4):
            true_wp.append([float(row.get(f'wp{i}_x', 'nan')), float(row.get(f'wp{i}_y', 'nan'))])
        true_wp = np.array(true_wp)
        print(f"Step {row['step']} true_wp: {true_wp}")
        # 检查shape
        if pred_wp.shape[0] < 4:
            print(f"[WARN] pred_wp shape异常: {pred_wp.shape}, 用0填充")
            pred_wp = np.pad(pred_wp, ((0, 4-pred_wp.shape[0]), (0,0)), mode='constant', constant_values=0)
        if true_wp.shape[0] < 4:
            print(f"[WARN] true_wp shape异常: {true_wp.shape}, 用0填充")
            true_wp = np.pad(true_wp, ((0, 4-true_wp.shape[0]), (0,0)), mode='constant', constant_values=0)
        wp_errs = np.linalg.norm(pred_wp[:4] - true_wp[:4], axis=1)
        print(f"Step {row['step']} wp_errs: {wp_errs}")
        if np.any(np.isnan(pred_wp)):
            print(f"[WARN] pred_wp 含有 nan，模型输出异常！")
        max_err = np.max(wp_errs)
        all_errs.append(wp_errs)
        pred_wps_list.append(pred_wp[:4])
        true_wps_list.append(true_wp[:4])
        if np.any(wp_errs > args.threshold):
            print(f"{step:>5} | {str(np.round(wp_errs,3)):<40} | {max_err:8.3f}")

    print('-'*60)
    all_errs = np.array(all_errs)
    print(f"Mean abs waypoint error: {np.mean(all_errs):.4f} m, Max error: {np.max(all_errs):.4f} m")

    if args.save_mp4:
        s, e = 0, len(pred_wps_list)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xlim(-10, 30)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('x (local)')
        ax.set_ylabel('y (local)')
        ax.set_title('PlanT Waypoint Prediction vs Ground Truth')
        pred_line, = ax.plot([], [], 'ro-', label='Pred Waypoints')
        true_line, = ax.plot([], [], 'bo-', label='True Waypoints')
        ax.legend()
        def update(i):
            pred = pred_wps_list[s+i]
            true = true_wps_list[s+i]
            pred_line.set_data(pred[:,0], pred[:,1])
            true_line.set_data(true[:,0], true[:,1])
            return pred_line, true_line
        ani = FuncAnimation(fig, update, frames=e-s, interval=100, blit=True)
        ani.save(args.save_mp4, writer='ffmpeg', fps=10)
        print(f"[INFO] 已保存所有{e-s}帧为mp4: {args.save_mp4}")

if __name__ == '__main__':
    main() 
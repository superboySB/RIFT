'''
PlanT PID 控制器独立测试模块
====================================

功能：
    - 读取 PlanT 生成的 pid_debug.csv（包含每步的两个waypoint、自车速度、以及真实控制量）
    - 用csv中的waypoint和ego_speed作为输入，调用PlanT工程内的PID控制器代码，输出预测的steer、throttle、brake
    - 将预测值与csv中的真实值（steer、throttle、brake）逐步对比，输出每步的预测值、真值和误差
    - 统计所有步的平均绝对误差，便于验证PID实现的正确性
    - 只打印误差大于阈值（THRESHOLD）的行，便于聚焦异常
    - 支持将指定step区间的PID控制过程保存为mp4动画，直观展示steer（方向）、throttle（油门）、brake（刹车）的预测与真实对比

参数说明：
    --csv : str, 必填。csv文件路径。例如 /workspace/RIFT/data/pid_debug.csv
    --pre_defined_desired_speed : float, 可选。PID控制器中的预设目标速度，默认8.0。
    --threshold : float, 可选。只打印任一控制量误差大于该阈值的行，默认0.01。
    --save_mp4 : str, 必填。保存动画的mp4文件路径，动画包含指定step区间，fps=10。
    --start : int, 可选。mp4输出起始step（含），默认0。
    --end : int, 可选。mp4输出终止step（不含），默认500。

用法示例：
    python rift/ego/plant/test_pid.py --csv /workspace/RIFT/data/pid_debug.csv --save_mp4 out.mp4 --start 0 --end 500

依赖说明：
    - 需已安装matplotlib和ffmpeg（系统需有ffmpeg命令）。
    - 不再弹窗显示，只保存mp4动画。

可视化内容：
    - 自车（原点）、两个waypoint、速度箭头
    - steer（方向盘）：预测/真实用不同颜色箭头
    - throttle（油门）/brake（刹车）：预测/真实用不同颜色的水平条直观显示
    - 详细图例和文字注释
'''
import csv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# ===== 可视化依赖与后端说明 =====
# 只保存mp4动画，不弹窗。需系统已安装ffmpeg。
import matplotlib
matplotlib.use('Agg')  # 只用于文件输出，无需GUI

# 复制自 model.py
class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        return self._K_P * error + self._K_I * integral + self._K_D * derivative

def control_pid(waypoints, velocity, turn_controller, speed_controller, pre_defined_desired_speed=8.0, is_stuck=False):
    waypoints = np.array(waypoints)
    waypoints[:, 0] += 1.3
    speed = velocity
    desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * pre_defined_desired_speed // 2
    if is_stuck:
        desired_speed = pre_defined_desired_speed
    brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
    delta = np.clip(desired_speed - speed, 0.0, 0.25)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, 0.75)
    throttle = throttle if not brake else 0.0
    aim = (waypoints[1] + waypoints[0]) / 2.0
    angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
    if brake:
        angle = 0.0
    steer = turn_controller.step(angle)
    steer = np.clip(steer, -1.0, 1.0)
    return steer, throttle, float(brake)

def visualize_pid(rows, pred_controls, true_controls, threshold=0.01, mp4_path=None):
    '''
    动态可视化PID控制过程，保存为mp4动画（所有帧，fps=10）。
    - mp4_path为str时保存mp4，不弹窗。
    '''
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-10, 30)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X (forward, m)')
    ax.set_ylabel('Y (left, m)')
    ax.set_title('PID Control Visualization (Ego at origin, facing +X)')
    ego_marker, = ax.plot([0], [0], 'ko', markersize=10, label='Ego')
    wp0_marker, = ax.plot([], [], 'bo', markersize=8, label='wp0')
    wp1_marker, = ax.plot([], [], 'go', markersize=8, label='wp1')
    arrows = []
    bar_patches = []
    text_objs = []  # 新增：用于存储每帧的数值文本对象
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    # 图例补充
    from matplotlib.patches import Patch
    legend_handles = [ego_marker,
                     Patch(color='b', label='wp0'),
                     Patch(color='g', label='wp1'),
                     Patch(color='c', label='Speed'),
                     Patch(color='r', label='Steer_pred'),
                     Patch(color='m', label='Steer_true'),
                     Patch(color='#66ff66', label='Throttle_pred'),
                     Patch(color='#006600', label='Throttle_true'),
                     Patch(color='#ff6666', label='Brake_pred'),
                     Patch(color='#990000', label='Brake_true')]
    ax.legend(handles=legend_handles, loc='upper right')
    def update(i):
        row = rows[i]
        steer_pred, thr_pred, brk_pred = pred_controls[i]
        steer_true, thr_true, brk_true = true_controls[i]
        wp0 = [float(row['wp0_x']), float(row['wp0_y'])]
        wp1 = [float(row['wp1_x']), float(row['wp1_y'])]
        ego_speed = float(row['ego_speed'])
        # 清除上帧内容
        for arr in arrows:
            arr.remove()
        arrows.clear()
        for bar in bar_patches:
            bar.remove()
        bar_patches.clear()
        for t in text_objs:
            t.remove()
        text_objs.clear()
        # Ego
        ego_marker.set_data([0], [0])
        # Waypoints
        wp0_marker.set_data([wp0[0]], [wp0[1]])
        wp1_marker.set_data([wp1[0]], [wp1[1]])
        # 速度箭头（x轴正方向，长度按ego_speed缩放）
        arr1 = ax.arrow(0, 0, ego_speed, 0, color='c', width=0.1, alpha=0.5, length_includes_head=True)
        arrows.append(arr1)
        # 预测steer箭头（以ego为圆心，方向为steer_pred*90度，长度固定）
        steer_angle_pred = steer_pred * 90
        arr2 = ax.arrow(0, 0, 5 * np.cos(np.radians(steer_angle_pred)), 5 * np.sin(np.radians(steer_angle_pred)), color='r', width=0.05, alpha=0.7, length_includes_head=True)
        arrows.append(arr2)
        # 真实steer箭头
        steer_angle_true = steer_true * 90
        arr3 = ax.arrow(0, 0, 5 * np.cos(np.radians(steer_angle_true)), 5 * np.sin(np.radians(steer_angle_true)), color='m', width=0.05, alpha=0.7, length_includes_head=True)
        arrows.append(arr3)
        # throttle/brake条形仪表
        # throttle: 绿色，brake: 红色，预测浅色，真实深色
        # throttle条
        bar_y0 = -15
        bar_len = 10
        bar_height = 1.2
        # 预测throttle
        bar_pred = ax.barh(bar_y0, thr_pred * bar_len, height=bar_height, left=0, color='#66ff66', alpha=0.7, label='Throttle_pred')
        bar_patches.extend(bar_pred)
        # 真实throttle
        bar_true = ax.barh(bar_y0 + bar_height, thr_true * bar_len, height=bar_height, left=0, color='#006600', alpha=0.7, label='Throttle_true')
        bar_patches.extend(bar_true)
        # brake条
        bar_y1 = -18
        # 预测brake
        bar_pred_b = ax.barh(bar_y1, brk_pred * bar_len, height=bar_height, left=0, color='#ff6666', alpha=0.7, label='Brake_pred')
        bar_patches.extend(bar_pred_b)
        # 真实brake
        bar_true_b = ax.barh(bar_y1 + bar_height, brk_true * bar_len, height=bar_height, left=0, color='#990000', alpha=0.7, label='Brake_true')
        bar_patches.extend(bar_true_b)
        # throttle/brake数值
        t1 = ax.text(bar_len + 1, bar_y0, f"{thr_pred:.2f}", va='center', ha='left', fontsize=10, color='#66ff66')
        t2 = ax.text(bar_len + 1, bar_y0 + bar_height, f"{thr_true:.2f}", va='center', ha='left', fontsize=10, color='#006600')
        t3 = ax.text(bar_len + 1, bar_y1, f"{brk_pred:.2f}", va='center', ha='left', fontsize=10, color='#ff6666')
        t4 = ax.text(bar_len + 1, bar_y1 + bar_height, f"{brk_true:.2f}", va='center', ha='left', fontsize=10, color='#990000')
        text_objs.extend([t1, t2, t3, t4])
        # 文字信息
        info = f"step: {row['step']}\n" \
               f"ego_speed: {ego_speed:.2f} m/s\n" \
               f"wp0: ({wp0[0]:.2f}, {wp0[1]:.2f})\n" \
               f"wp1: ({wp1[0]:.2f}, {wp1[1]:.2f})\n" \
               f"steer_pred: {steer_pred:.3f}, steer_true: {steer_true:.3f}\n" \
               f"throttle_pred: {thr_pred:.3f}, throttle_true: {thr_true:.3f}\n" \
               f"brake_pred: {brk_pred:.3f}, brake_true: {brk_true:.3f}"
        text_box.set_text(info)
        return ego_marker, wp0_marker, wp1_marker, text_box, *arrows, *bar_patches, *text_objs
    frames = len(rows)
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False, repeat=False)
    if mp4_path:
        ani.save(mp4_path, writer='ffmpeg', fps=10)
        print(f"[INFO] 已保存所有{frames}帧为mp4: {mp4_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test PlanT PID controller with CSV input')
    parser.add_argument('--csv', type=str, required=True, help='Path to pid_debug.csv')
    parser.add_argument('--pre_defined_desired_speed', type=float, default=8.0, help='Desired speed used in PID')
    parser.add_argument('--threshold', type=float, default=0.01, help='Only print rows with abs error above this threshold')
    parser.add_argument('--save_mp4', type=str, required=True, help='Path to save mp4 animation (all steps, fps=10)')
    parser.add_argument('--start', type=int, default=0, help='Start step for mp4 output (inclusive), default 0')
    parser.add_argument('--end', type=int, default=500, help='End step for mp4 output (exclusive), default 500')
    # 用法示例：
    # python rift/ego/plant/test_pid.py --csv /workspace/RIFT/data/pid_debug.csv --save_mp4 out.mp4 --start 0 --end 500
    args = parser.parse_args()
    turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
    speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    pred_controls = []
    true_controls = []
    print(f"{'step':>5} | {'steer_pred':>10} | {'steer_true':>10} | {'thr_pred':>8} | {'thr_true':>8} | {'brk_pred':>8} | {'brk_true':>8} | {'steer_err':>9} | {'thr_err':>8} | {'brk_err':>8}")
    print('-'*110)
    steer_errs, thr_errs, brk_errs = [], [], []
    for row in rows:
        try:
            wp0 = [float(row['wp0_x']), float(row['wp0_y'])]
            wp1 = [float(row['wp1_x']), float(row['wp1_y'])]
            ego_speed = float(row['ego_speed'])
            steer_true = float(row['steer'])
            thr_true = float(row['throttle'])
            brk_true = float(row['brake'])
            waypoints = np.array([wp0, wp1])
            steer_pred, thr_pred, brk_pred = control_pid(waypoints, ego_speed, turn_controller, speed_controller, args.pre_defined_desired_speed)
            steer_err = abs(steer_pred - steer_true)
            thr_err = abs(thr_pred - thr_true)
            brk_err = abs(brk_pred - brk_true)
            steer_errs.append(steer_err)
            thr_errs.append(thr_err)
            brk_errs.append(brk_err)
            pred_controls.append((steer_pred, thr_pred, brk_pred))
            true_controls.append((steer_true, thr_true, brk_true))
            if steer_err > args.threshold or thr_err > args.threshold or brk_err > args.threshold:
                print(f"{row['step']:>5} | {steer_pred:10.4f} | {steer_true:10.4f} | {thr_pred:8.4f} | {thr_true:8.4f} | {brk_pred:8.4f} | {brk_true:8.4f} | {steer_err:9.4f} | {thr_err:8.4f} | {brk_err:8.4f}")
        except Exception as e:
            print(f"Error in step {row.get('step', '?')}: {e}")
    print('-'*110)
    print(f"Mean abs error: steer={np.mean(steer_errs):.4f}, throttle={np.mean(thr_errs):.4f}, brake={np.mean(brk_errs):.4f}")
    # 保存mp4动画（指定step区间）
    s, e = args.start, args.end
    visualize_pid(rows[s:e], pred_controls[s:e], true_controls[s:e], threshold=args.threshold, mp4_path=args.save_mp4)

if __name__ == '__main__':
    main() 
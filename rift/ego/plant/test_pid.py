'''
PlanT PID 控制器独立测试模块
====================================

功能：
    - 读取 PlanT 生成的 pid_debug.csv（包含每步的两个waypoint、自车速度、以及真实控制量）
    - 用csv中的waypoint和ego_speed作为输入，调用PlanT工程内的PID控制器代码，输出预测的steer、throttle、brake
    - 将预测值与csv中的真实值（steer、throttle、brake）逐步对比，输出每步的预测值、真值和误差
    - 统计所有步的平均绝对误差，便于验证PID实现的正确性
    - 只打印误差大于阈值（THRESHOLD）的行，便于聚焦异常

参数说明：
    --csv : str, 必填。csv文件路径。例如 /workspace/RIFT/data/pid_debug.csv
    --pre_defined_desired_speed : float, 可选。PID控制器中的预设目标速度，默认8.0。
        该值在PlanT工程的model.py中，CarlaDataProvider.get_desired_speed() 默认返回8.0（单位m/s），
        这是PlanT默认的巡航速度设定。
    --threshold : float, 可选。只打印任一控制量误差大于该阈值的行，默认0.01。

用法示例：
    python rift/ego/plant/test_pid.py --csv /workspace/RIFT/data/pid_debug.csv
    # 或自定义目标速度和误差阈值
    python rift/ego/plant/test_pid.py --csv /workspace/RIFT/data/pid_debug.csv --pre_defined_desired_speed 8.0 --threshold 0.02

输出说明：
    - 只输出误差大于阈值的行：step号、PID预测/真值、各自误差
    - 最后输出所有步的平均绝对误差

注意：
    - 本程序严格复现PlanT工程内的PID控制器实现，输入输出与原csv完全一致。
    - 你可以直接用csv文件（如 /workspace/RIFT/data/pid_debug.csv）进行验证。
'''
import csv
import numpy as np
from collections import deque

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test PlanT PID controller with CSV input')
    parser.add_argument('--csv', type=str, required=True, help='Path to pid_debug.csv')
    parser.add_argument('--pre_defined_desired_speed', type=float, default=8.0, help='Desired speed used in PID')
    parser.add_argument('--threshold', type=float, default=0.01, help='Only print rows with abs error above this threshold')
    args = parser.parse_args()

    turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
    speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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
            if steer_err > args.threshold or thr_err > args.threshold or brk_err > args.threshold:
                print(f"{row['step']:>5} | {steer_pred:10.4f} | {steer_true:10.4f} | {thr_pred:8.4f} | {thr_true:8.4f} | {brk_pred:8.4f} | {brk_true:8.4f} | {steer_err:9.4f} | {thr_err:8.4f} | {brk_err:8.4f}")
        except Exception as e:
            print(f"Error in step {row.get('step', '?')}: {e}")
    print('-'*110)
    print(f"Mean abs error: steer={np.mean(steer_errs):.4f}, throttle={np.mean(thr_errs):.4f}, brake={np.mean(brk_errs):.4f}")

if __name__ == '__main__':
    main() 
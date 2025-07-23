# PlanT PID 控制器实现说明

本报告严格基于 `rift/ego/plant/model.py` 和 `rift/ego/plant/test_pid.py` 的代码实现，详细说明 PlanT 工程中 PID 控制器的输入、输出、核心公式、误差定义及实现细节。

---

## 1. 控制器输入与输出

- **输入**：
  - 当前自车速度 $v_{ego}$（单位：m/s）
  - 预测轨迹的前两个waypoint $wp_0 = (x_0, y_0)$，$wp_1 = (x_1, y_1)$（均为局部坐标）
- **输出**：
  - 方向盘转角 $steer$（范围 $[-1, 1]$）
  - 油门 $throttle$（范围 $[0, 0.75]$）
  - 刹车 $brake$（范围 $[0, 1]$，实际为0或1）

---

## 2. 速度控制（throttle/brake）

### 2.1 目标速度的计算

$$
\text{desired\_speed} = \|wp_1 - wp_0\| \times \frac{v_{set}}{2}
$$

其中 $v_{set}$ 为全局预设目标速度（代码默认8.0 m/s）。

### 2.2 油门/刹车判据

- 若 $\text{desired\_speed} < 0.4$ 或 $\frac{v_{ego}}{\text{desired\_speed}} > 1.1$，则 $brake = 1$，$throttle = 0$
- 否则：
  - 速度误差 $\Delta v = \text{clip}(\text{desired\_speed} - v_{ego}, 0, 0.25)$
  - $throttle = \text{PID}_{speed}(\Delta v)$，再clip到$[0, 0.75]$

---

## 3. 方向控制（steer）

- 计算目标方向：
  $$
  aim = \frac{wp_0 + wp_1}{2}
  $$
  $$
  angle = \frac{\arctan2(aim_y, aim_x) \times 180}{90\pi}
  $$
- 若 $brake = 1$，则 $angle = 0$
- $steer = \text{PID}_{turn}(angle)$，再clip到$[-1, 1]$

---

## 4. PID 控制器公式

对于任意误差 $e$，PID输出：
$$
PID(e) = K_P \cdot e + K_I \cdot \text{mean}(e_{hist}) + K_D \cdot (e_{curr} - e_{prev})
$$
- $K_P, K_I, K_D$ 分别为比例、积分、微分系数
- $e_{hist}$为历史误差窗口（长度20）

参数：
- 速度控制：$K_P=5.0, K_I=0.5, K_D=1.0$
- 转向控制：$K_P=1.25, K_I=0.75, K_D=0.3$

---

## 5. 误差定义与评估

- 逐步对比 $steer, throttle, brake$ 的预测值与csv真值，统计平均绝对误差：
  $$
  \text{MAE}_{steer} = \frac{1}{N} \sum_{i=1}^N |steer_{pred,i} - steer_{true,i}|
  $$
  其余同理。

---

## 6. 代码实现的特殊点

- 只用前两个waypoint和当前速度，无需假设waypoint间的时间间隔。
- 目标速度用空间距离近似，不做物理速度严格推算。
- brake为0/1，throttle最大0.75。
- 误差窗口长度20，积分项为窗口均值。

---

## 7. 适用范围与局限性

- 该PID实现适用于PlanT工程的轨迹跟踪与控制仿真。
- 若需物理上更严谨的速度控制，应考虑waypoint间的时间间隔。
- 该实现与PlanT原始代码完全一致，适合复现与对比实验。 
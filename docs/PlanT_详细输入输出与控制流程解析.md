# PlanT 模型详细输入输出与控制流程解析

---

## 1. PlanT 是什么？
PlanT 是一种基于结构化拓扑信息的自动驾驶决策模型，其输入为感知/仿真环境中提取的车辆、路线等结构化要素（非原始视觉数据），输出为未来一段时间的轨迹点（waypoints），再通过轨迹跟踪控制器（如PID）转为底层控制指令（油门、刹车、方向盘），最终驱动车辆。

---

## 2. PlanT 的输入是什么？

### 2.1 输入来源
- **非视觉端到端**：PlanT 不直接处理BEV图像或相机图像。
- **输入为结构化拓扑信息**，即：
  - 自车及周围车辆的状态（位置、速度、朝向、尺寸等）
  - 路径/路线点（Route）的结构化信息
  - 交通灯等环境要素（如有）

### 2.2 输入特征构建（详见 `plant_agent.py:get_input_batch`）
- **车辆特征（data_car）**：
  - `[type_indicator, x_relative, y_relative, yaw, speed, extent_x, extent_y]`
    - type_indicator: 1.0（车辆）
    - x/y_relative: 车辆相对自车的坐标
    - yaw: 朝向（角度）
    - speed: 速度（km/h）
    - extent_x/y: 车辆尺寸
- **路线特征（data_route）**：
  - `[type_indicator, x_relative, y_relative, yaw, id, extent_x, extent_y]`
    - type_indicator: 2.0（路线点）
    - id: 路线点编号
- **目标点（target_point）**：
  - 由全局规划器提供，作为轨迹预测的目标
- **交通灯状态（light_hazard）**：
  - 作为辅助输入

### 2.3 输入数据流
- 所有特征拼接后，打包成张量，送入Transformer模型
- **无原始视觉感知信息**，输入为“感知后的拓扑结构真值”

---

## 3. PlanT 的输出是什么？

### 3.1 输出内容
- **输出为未来一段时间的轨迹点（waypoints）**
  - 形状为 `[batch, pred_len, 2]`，每行为未来某一时刻的 (x, y) 位置（自车坐标系下）
- 由模型的 `forward` 方法生成

### 3.2 关键代码
- `plant_agent.py:_get_control`：
  ```python
  _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)
  ```
- `model.py:forward`：
  - Transformer编码结构化输入，输出未来轨迹点

---

## 4. 轨迹如何转为底层控制（油门、刹车、方向盘）？

### 4.1 控制器原理
- **PlanT 内置 PID 控制器**，根据当前速度和预测轨迹点，计算出目标速度、目标方向
- 输出为：
  - `steer`（方向盘）
  - `throttle`（油门）
  - `brake`（刹车）

### 4.2 关键代码
- `plant_agent.py:_get_control`：
  ```python
  steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)
  ```
- `model.py:control_pid`：
  ```python
  def control_pid(self, waypoints, velocity, is_stuck=False):
      # 轨迹点转为控制量
      ...
      return steer, throttle, brake
  ```

### 4.3 详细流程
1. **模型输出轨迹点**：`pred_wp`
2. **调用 `control_pid` 方法**：
   - 输入：`waypoints`（轨迹点），`velocity`（当前速度）
   - 处理：用PID控制器根据当前速度和轨迹点，计算目标速度、目标方向
   - 输出：`steer`、`throttle`、`brake`
3. **封装为 `carla.VehicleControl`**，下发到仿真环境

---

## 5. 代码调用链梳理

1. **环境调用 PlanT policy 的 get_action**（`plant.py`）
   - `control = self.planner_list[info['env_id']].run_step(obs[i])`
2. **PlanTAgent.run_step**（`plant_agent.py`）
   - 处理输入，生成结构化特征
   - `self._get_control(label_raw, tick_data)`
3. **PlanTAgent._get_control**
   - `x, y, _, tp, light = self.get_input_batch(label_raw, input_data)`
   - `_, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)`
   - `steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)`
   - 封装为 `carla.VehicleControl` 下发
4. **HFLM.forward**（`model.py`）
   - Transformer编码结构化输入，输出未来轨迹点
5. **HFLM.control_pid**（`model.py`）
   - 轨迹点+当前速度 → PID → steer/throttle/brake

---

## 6. 与视觉端到端方法的对比

- **PlanT**：输入为结构化拓扑信息（车辆/路线/环境要素），输出轨迹点，再转为底层控制。
- **端到端视觉方法（如UniAD）**：输入为BEV图像或相机图像，直接输出轨迹或底层控制。
- **PlanT 的优势**：感知-决策解耦，便于解释和调试，适合仿真/真值环境。

---

## 7. 关键文件与函数索引

- `rift/ego/plant/plant.py`：PlanT policy主入口
- `rift/ego/plant/plant_agent.py`：PlanTAgent，输入输出与控制主流程
- `rift/ego/plant/lit_module.py`：LitHFLM，模型封装
- `rift/ego/plant/model.py`：HFLM，模型本体与PID控制器

---

## 8. 总结

- **PlanT 输入**：结构化车辆/路线/环境要素信息（无视觉原始数据）
- **PlanT 输出**：未来一段时间的轨迹点（waypoints）
- **轨迹转控制**：通过内置PID控制器，将轨迹点转为油门、刹车、方向盘指令，最终下发到仿真环境
- **全流程代码已梳理，所有关键细节如上**

如需进一步追踪某一环节的具体实现或参数细节，可继续指定！ 

---

## 9. 轨迹输出到底层控制的“白盒性”说明

PlanT从轨迹输出到底层控制的全流程如下：

1. **模型输出轨迹点**：PlanT模型（`model.py:HFLM.forward`）输出未来一段时间的轨迹点（`pred_wp`）。
2. **白盒PID轨迹跟踪**：`model.py:HFLM.control_pid` 使用完全开源的PID控制器（`PIDController`类），根据轨迹点和当前速度，计算出油门、刹车、方向盘指令。所有控制算法均为白盒实现，参数和逻辑可追溯、可修改，无任何CARLA黑盒调用。
3. **控制指令封装与下发**：`plant_agent.py:_get_control` 将上述控制量封装为 `carla.VehicleControl` 对象，仅作为数据结构传递给CARLA仿真环境，由CARLA负责物理执行。此过程不涉及任何黑盒决策或轨迹跟踪逻辑。

**结论**：PlanT的轨迹跟踪与底层控制完全为白盒实现，CARLA仅作为执行器接收控制指令，无任何黑盒轨迹跟踪或控制逻辑参与。 
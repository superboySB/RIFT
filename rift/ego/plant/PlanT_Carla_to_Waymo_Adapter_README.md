# PlanT模型从Carla到Waymo Motion 1.2数据集的适配手册

---

## 1. Carla环境下PlanT模型输入输出结构

### 输入（每一步）
- **Ego车辆状态**：x, y, yaw, speed, extent_x, extent_y, extent_z, id
- **Other Vehicles（最多20辆）**：每辆车7个特征（x, y, yaw, speed, extent_x, extent_y, id）
- **Route Points（最多3个）**：每个点6个特征（x, y, yaw, id, extent_x, extent_y）
- **Target Point**：x, y（通常为局部规划目标点）
- **Light Hazard**：交通灯危险标志（0/1）

### 输出（每一步）
- **Waypoints（4个）**：每个点2个特征（x, y），为未来轨迹点（自车坐标系下）

---

## 2. Waymo Motion 1.2下可获得的输入输出结构

### 可直接获得
- **Ego车辆状态**：可从自车历史/未来轨迹获得x, y, heading（yaw），速度，车辆尺寸（extent），id
- **Other Vehicles**：同上，Waymo提供所有参与体的历史/未来轨迹和尺寸
- **Target Point**：可用未来轨迹的终点或某一采样点
- **交通灯状态**：Waymo提供交通灯状态（但无直接hazard标志）

### 需额外处理/推算
- **Route Points**：Waymo不直接提供“route point”，需用地图元素（如车道中心线）+自车轨迹推算
- **Extent（Route Point）**：Waymo地图元素无包围盒尺寸，可设为0或1
- **Light Hazard**：需自定义（如红灯为1，其它为0）

### 输出
- **Waypoints**：PlanT模型输出，与Carla一致

---

## 3. 适配/额外处理事项

### 3.1 Route Points生成
- 方案1：用自车未来轨迹在地图上投影，采样3个点作为route point，字段为x, y, yaw（heading），id（采样顺序），extent_x/extent_y设为0或1。
- 方案2：用车道中心线（lane centerline）采样3个点，字段同上。
- 字段补全：extent_x/extent_y/extent_z可统一设为0或1。

### 3.2 Light Hazard定义
- Waymo交通灯有状态（红/黄/绿），可自定义hazard：如红灯为1，其它为0。

### 3.3 其它字段补全
- id：可用Waymo object id或采样顺序编号
- extent：Waymo车辆有尺寸，直接填入；route point无尺寸，补0或1

### 3.4 坐标系
- 建议统一为自车坐标系（如Carla），如需转换需做坐标变换

---

## 4. 迁移注意事项和建议
- **输入字段需严格对齐**，缺失字段需补0/1/NaN
- **route point生成策略需统一**，建议采样自车轨迹或车道中心线
- **交通灯hazard需自定义**，与Carla一致
- **坐标系需统一**，建议全部转为自车坐标系
- **输出waypoints与Carla一致**，无需更改

---

## 5. 适配流程简要步骤

1. **解析Waymo数据集**，提取自车及周围车辆的状态、轨迹、尺寸、交通灯状态
2. **生成route points**：
   - 采样自车未来轨迹或车道中心线，生成3个点，补全字段
3. **生成target point**：
   - 取未来轨迹终点或某一采样点
4. **定义light hazard**：
   - 红灯为1，其它为0
5. **补全所有输入字段**，对齐Carla输入格式
6. **将输入送入PlanT模型，收集输出waypoints**
7. **保存输入输出为csv**，便于后续离线评测

---

如需具体代码示例或采样策略实现，可进一步补充需求！ 
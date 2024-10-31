
## I. Classification
- Perception环境感知
-   - 目标检测算法：YOLO, R-CNN
    - SLAM
- Position
- Planning决策规划
-   - 行为决策算法：通常基于有限状态机或强化学习
- Control
-   - PID
    - MPC

## 实战项目
- 交通标识牌检测
- 行人检测
- 车辆检测
- 3D目标检测
- 红绿灯检测
- 斑马想检测与识别

## 论文领域分类
- 自动驾驶
- 3D目标检测
- 3D语义分割
- 车辆重识别&车辆检测
- 车道轨迹预测
- 三维重建
- 行人轨迹预测
- 深度估计
     - Loss损失函数定义：
       
     $D(y, y^{*}) = \frac{1}{2n^2}\sum_{i, j}((logy_i-logy_j)-(logy_i^*-logy_j^*))^2$
-     
- ...
## 开源项目
- Donkeycar
- Autoware
- OpenPilot
- AirSim
- DriveLM
  
## Sensors
基于纯视觉Camera的方案成本要比基于Lidar的方案成本低10倍
1. Camera: 图像特征：纹理丰富，成本低，基于图像的基础模型相对成熟和完善，较容易扩展到BEV感知算法
2. Lidar: 点云的特征：稀疏性、无序性、是一种3D表征、远少近多

## End-to-end
- 传统自动驾驶

sensor $\rightarrow$ perception  $\rightarrow$ prediction  $\rightarrow$  planning  $\rightarrow$ control  $\rightarrow$ executor

缺点：
- 模块化之间累积误差
- 模块化之间优化目标不一致
- 模块化之间计算冗余
- 模块局部最优不等于全局最优（存疑）
- 基于规则的决策系统复杂庞大，较难维护
  
- 端到端自动驾驶（LLMs如何赋能AD）

sensor $\rightarrow$ end-to-end AD model $\rightarrow$ control  $\rightarrow$ executor

特点：
- 自动驾驶第一性原理（？）
- 避免级联误差
- 消除计算冗余
- 从规则驱动 $\rightarrow$ 数据驱动

缺点：
- 黑盒，可解释性
- 因果倒置？
- 
## II. Perception
- BEV(Bird's-Eye-view)
  - 优势：
  - 没有前视图近大远小的透视畸变效果
  - 没有遮挡
  - 什么是BEV表征
    - 是一个重构视角
    - 是一个多传感器融合的空间
    - 是固定视角（俯视视角）的空间
- BEVFormer(BEV Camera - 仅依赖于相机)
-   - Structure：Multi-view input -> Backbone -> Multi-Camera features
    - 2 attention: Temporal Attention & Spatial Attention
- BEVFusion(同时依赖于相机输入和点云数据)
-   - 其中图像处理模块的神经网络与BEVFormer中是一致的
- Fusion
  - Point-level Fusion
  - Feature-level Fusion
  均从点云数据出发，去查询图像数据对应点或特征。依赖于点云数据的准确性和充分性。
  - BEVFusion：在图像和点云的BEV feature层面上进行融合
  无主次关系
    - camera stream
    - LiDAR stream
    -   - backbone: pointpillar, center-piont, transfusion
  

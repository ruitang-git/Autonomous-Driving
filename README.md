
## I. Classification
- HD(High-Definition) Map高精地图
- Positioning
- Perception环境感知
-   - 目标检测算法：YOLO, R-CNN
    - SLAM
- Prediction
(PNC: Planning & Control)
规划基础知识：Search-based methods;Sampling-based methods; Kinematics-based methods; Optimization-based methods(main stream)
- Planning决策规划
    - 分类：Decoupling(解耦) and Coupling(联合)
    - Decoupling
    -     - longitudinal-lateral decomposition
    -         - Optimal trajectory generation for dynamic street scenarios in a frenet frame
    -     - path-speed decompostion
    -         - Baidu apollo em motin planner
    - Coupling
    -     - Behavior Planning
    -     - Motion Planning    
-   - 行为决策算法：通常基于有限状态机或强化学习
- Control
-   - Geometric-based Path Tracking Methods
-   - PID
    - Linear Quadratic Regulator (LQR)
    - Model Predictive Control(MPC) 

## HD Map
1. 精度：cm级；传统：m级
2. 高精度地图可以用于地位（类似于拼图，在图上找出自己所在的位置，根据传感器的数据与高精度地图的数据进行匹配）
3. 用于感知（ROI: Region of Interest, 高精地图给出某些先验；恶劣环境传感器信号受影响时...)
4. 用于规划（提供减速带，限速，红绿灯信息...）
## Positioning
1. GNSSRTK
GNSS全称为Global Navigation Satellite System，GPS为其中一种。GNSS要实现定位，至少需要接收器收到3颗卫星的信号，一般保证地面上的接收器能在任何位置收到4颗卫星的信号。由于光速传播速度很快，时延上测量的一些微小误差会被极大的放大，因此每个卫星都配备了原子钟。同时借助Real-Time Kinematic(RTK), 在地面上建立几个基站，每个基站都知道自己的真实位置，同时也通过GPS进行位置测量，这两者之间的误差也会传播给GPS接收器用于校准。在RTK的加持下，GPS的定位精度可达10cm。GPS不足以支撑自动驾驶的定位的两个原因为1. 在城市和遮挡道路下信号收到严重阻碍 2. 更新频率低，约为10Hz。
2. 惯性导航IMU
IMU 通常由加速度计、陀螺仪和磁力计等传感器组成。
加速度计：用于测量物体在三个坐标轴上的加速度。通过对加速度的积分，可以得到物体的速度和位移信息。
陀螺仪：用于测量物体在三个坐标轴上的角速度。陀螺仪可以提供物体的旋转信息，帮助确定物体的姿态和方向。
磁力计：用于测量地球磁场的强度和方向。磁力计可以提供物体的方位信息，帮助确定物体的朝向。
优点在于更新频率很高，可到1000Hz，缺点在于误差会随时间累积。因此适合短时间内的实时定位。可以使用GPS和IMU融合定位，可以解决GPS更新频率低的问题，但是信号遮挡的问题仍然存在。
卡尔曼滤波
4. 激光雷达定位
通过将点云数据和高精地图匹配得到，依赖于高精度地图
5. 视觉定位
粒子滤波
## 实战项目
- 交通标识牌检测
- 行人检测
- 车辆检测
- 3D目标检测
- 红绿灯检测
- 斑马想检测与识别

## 论文领域分类
- 3D目标检测
- 3D语义分割Semantic segmentation：按照类别来进行区分
-     - 数据集：PASCAL VOC, CityScapes(自动驾驶数据集)，ADE20K
-     - 经典算法
-         - FCN
-         - ParseNet(ICLR 2016)
-         - GCN: Global Convolutional Network(CVPR 2017)
-         - PSPNet(CVPR 2017)
-         - Transformer-based: SETR(CVPR 2021), SegFormer(NIPS 2021), Swin-Unet, TransUNet, MaskFormer
-     - reference: [语义分割三十年(-2023)](https://www.bilibili.com/video/BV1xL41117SC/?spm_id_from=333.999.0.0&vd_source=d7339479a3c0de1eb46f840baa9a3510)
- 目标跟踪
- 三维重建
- 多模态/多传感器融合
- 深度估计 Depth Evaluation
     - Loss损失函数定义：
     - 
$\begin{equation}       
D(y, y^*) = \frac{1}{2n^2}\sum_{i, j}((logy_i-logy_j)-(logy_i^*-logy_j^*))^2
=\frac{1}{n}\sum_id_i^2-\frac{1}{n^2}\sum_{i,j}d_id_j
\end{equation}$
损失函数增加第二项的目的在于保证每个点和真实值的偏差必须为统一方向，比起所有点在真实值附近的一个小范围内波动，更倾向于使所有的估计点均比真实值小一个固定数值。
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
- 四个核心任务：Dectection检测，Classification分类，Tracking跟踪，Segmentation语义分割
-     - Detection&Classification: R-CNN, Fast(er) R-CNN, YOLO, SSD
-     - Tracking: 优势：1.可以解决Detection中的遮挡问题 2.保留身份preserver identity 流程：1.matching匹配 2.prediction预测
-     - Segmentation：确定车辆可驾驶区域，依赖于特殊的CNN:FCN, 网络中的所有层均为卷积层
- BEV(Bird's-Eye-view)
  - 优势：
  - 没有前视图近大远小的透视畸变效果
  - 没有遮挡
  - 什么是BEV表征
    - 是一个重构视角
    - 是一个多传感器融合的空间
    - 是固定视角（俯视视角）的空间
  - 公司
  -     特斯拉（带火），地平线，百度，滴滴
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

## Prediction
Model-based and Data-based
1. 基于车道的预测(关注车子在划分的若干个车道区域的转换过程)
2. 障碍物状态
3. 递归神经网络RNN??
4. 轨迹生成（基于多项式）

## Planning
1. route路由：high level
2. trajactory generation轨迹生成：low level，需要应对路由中行人，车辆等各种影响因素

坐标系转换：
Cartesian Coordinates $\rightarrow$ Frenet Coordinates(描述了汽车相对于道路的坐标)
Frenet Coordinates: s:代表沿道路的距离Longitudinal Axis; d:表示与纵向线的距离Lateral Axis

两种解耦的规划方案：
path-speed decompostion路径速度解耦规划
1. 路径规划：使用成本函数对候选路径进行筛选评估, 成本函数可包括与车道中心的距离，与障碍物的距离，曲率等；
通过将道路划分为多个单元格，并且在单个单元格内生成多个点，这些点的组合便可构建成多个候选路径；
2. 速度规划：与路径点相关的一系列速度（速度曲线speed profile)
ST图（用于构建速度曲线）： S：车辆的纵向位移 t:时间； 速度为曲线的斜率；障碍物可以在ST图中用图形表示，车辆曲线不得与其相交

Quadratic Programming: 将离散的路径平滑化

longitudinal-lateral decomposition
1. ST: 具有时间戳的纵向轨迹（S为Frenet Coordinates中的s量）
2. SL：相对于纵向轨迹的横向偏移
分别生成ST和SL轨迹，然后进行合并

## Control

## 一些资料
https://github.com/ProgramTraveler/Road-To-Autonomous-Driving?tab=readme-ov-file
  

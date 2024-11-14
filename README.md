
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
- 四个核心任务：Dectection检测，Classification分类，Tracking跟踪，Segmentation分割(普通分割，语义分割semantic segmentation，实例分割instance segmentation)
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
path-speed decompostion路径速度解耦规划（较主流）
1. 路径规划：使用成本函数对候选路径进行筛选评估, 成本函数可包括与车道中心的距离，与障碍物的距离，曲率等；
通过将道路划分为多个单元格，并且在单个单元格内生成多个点，这些点的组合便可构建成多个候选路径；
2. 速度规划：与路径点相关的一系列速度（速度曲线speed profile)
ST图（用于构建速度曲线）： S：车辆的纵向位移 t:时间； 速度为曲线的斜率；障碍物可以在ST图中用图形表示，车辆曲线不得与其相交

Quadratic Programming: 将离散的路径平滑化

longitudinal-lateral decomposition就（采样空间过大，一般自动驾驶不用。但适合物流小车等有限场景）
1. ST: 具有时间戳的纵向轨迹（S为Frenet Coordinates中的s量）
2. SL：相对于纵向轨迹的横向偏移
分别生成ST和SL轨迹，然后进行合并。
为降低采用空间，分为三个场景进行采样：
    - cruise
    - following  
    - stop

## Control
1. PID(Proportional-Integral-Derivative)
比例控制（P）：根据当前误差的大小成比例地调整控制输出。误差越大，控制输出的调整幅度就越大。比例控制的作用是快速响应系统的偏差，但不能消除静态误差。
积分控制（I）：对误差进行积分，随着时间的积累，积分项会增大，从而消除静态误差。但是积分控制可能会导致系统响应变慢，甚至出现超调。
微分控制（D）：根据误差的变化率来调整控制输出。微分控制可以预测系统的变化趋势，提前进行调整，从而提高系统的稳定性和响应速度。但微分控制对噪声比较敏感
优点：简单，且大部分情况很有效
缺点：(a)线性; (b)需要不同的PID控制器来控制转向和加速，难以将横向和纵向控制结合起来; (c)依赖于实时的误差测量
2. LQR(Linear-Quadratic Regulator)线性二次调节器
基于模型的控制器，使用车辆的状态来使误差最小化
e.g., Apollo使用LQR进行横向控制，横向控制包括四个组件(a)横向误差(b)横向误差的变化率(c)朝向误差(d)朝向误差的变化率，同时具有三个控制输入：(a)转向(b)加速(c)制动
L(linear) part: 适用于线性系统
$\begin{equation}
\cdot{x} = Ax+Bu
\end{equation}$
Q(Quadratic) part:
cost function:
$\begin{equation}
cost = \int_{0}^{\infty}(x^TQx+u^TRu)dt
\end{equation}$
通过最小化成本函数进行求解，控制方法被描述成$u=-Kx$, 即为求解K
3. MPC(Model Predictive Control)
分为三步：
- 建立车辆模型
- 利用优化引擎计算有限时间范围内的控制输入(计算未来的输入序列)
对于规划路径，考虑更长时间范围的控制输入，输出的结果更准确，相比之下短时间内的输入预测会导致输出路径的振荡，缺点是响应更慢。
- 执行第一组控制输入(只执行序列中的第一步)
优势：建立了车辆模型，因此比PID更精确

## ---

## Perception
### Segmentation
- 传统
- 基于深度学习
     - reference
         - A Survey on DNN based Semantic Segmentation
         - https://blog.csdn.net/Julialove102123/article/details/80493066
     - 通用框架
           - 前端：使用FCN进行特征提取
               - 下采样+上采样
                - 多尺度特征融合
                - 获得像素级别的segmentation map
           - 后端：使用CRF(条件随机场)/MRF(马尔可夫随机场)优化前端的输出得到优化后的分割图
    - 经典论文/模型
    -     - Fully Convolutional Networks(FCN)
    -     神经网络用作语义分割的开山之作。（a）提出了全卷积网络（全连接层替换为卷积层）（b）使用了反卷积层（从特征图映射为原图大小，上采样的功能）
    -     - DeepLab
    -     引入了空洞卷积
    -     - Pyramid Scene Parsing Network(PSPNet: CVPR 2017)
    -     核心贡献为Global Pyramid Pooling，将特征图缩放到几个不同的尺寸，使得特征具有更好的全局和多尺度信息
    -     - Mask R-CNN(BY Kaiming He: ICCV 2017)
    -     - U-Net(2015)
      
### Object Detection(目标检测)
#### PART I: 2D(i.e., 基于图片)
👉[code](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
- 算法流程
-     - 位置：先找到所有的ROI(Region of Interest)
-         - method1: Sliding Window(会产生很多无效的框，计算复杂)
-         - method2: Region Proposal
-             - 使用分割segmentation去做
-             - 减小了无效的框
-             - selective search：over segmentation, merge adjacent boxes according to similarity[https://www.learnopencv.com/selective-search-for-object-detection-cpp-python]
-         - method3: CNN
-             - anchor box
-             - RPN(from Faster-CNN)
-     - 类别：对每一个ROI做分裂获取类别信息
-     - 位置修正
- 算法分类
    - 基于图片的检测
         - RCNN(首次将CNN引入目标识别, Selective Search+CNN+SVM+linear regression)
         - SPPNet(在RCNN基础上实现CNN共享降低计算量，同时引入SPP层将不同尺寸的特征转换为固定尺寸的特征后接入全连接层)
         - Fast-RCNN(在SPPNet基础上分类和回归都用CNN实现，同时SPP层替换为ROI Pooling加速计算。但候选框还是基于Selective Search，未实现完全端到端)
         - Faster-RCNN(使用CNN提取候选框，即RPN(Region Proposal Net), 实现端到端训练)
    **以上算法都是基于预选框加分类回归的，称之为two-step；以下介绍one-step的算法；一般来说one-step精度会稍低，但实时性更好**
         - YOLO(全图划分成7x7网格，每个网格对应有2个default box)  DarkNet
         - SSD
         - YOLO-v2
         -     - 更丰富的default box
         -     - 更灵活的类别预测（把预测类别的机制从空间位置（cell）中解耦，由default box同时预测类别和坐标，有效解决物体重合？？？）
         - YOLO-v3
         -     - 更好的基础网络
         -     - 考虑多尺度
    - 基于点云的检测
#### PART II: 3D(i.e., 基于激光雷达点云)
- 传统vs深度学习
    - 传统：
          - 基于点云的目标检测： 分割地面$\rightarrow$点云聚类$\rightarrow$特征提取$\rightarrow$分类
          - 地面分割依赖于认为设计的特征和规则，如设置一些阈值，表面法线等
    - 深度学习：
          - 非结构化数据
          - 无序性
          - 数据稀疏
      
- Pixel-based
      - 基本思想：
          - 3D$\rightarrow$2D, 三维点云在不同角度的相机投影
          - 再借助2D图像领域的深度学习进行分析
      - 典型算法：
          - MVCNN(Multi-View CNN)
          - MV3D: 输入为BV+FV+RGB
          - AVOD: 输入为BEV+RGB
          - SqueezeSeg: 投影到球面
- Voxel-based
      - 基本思想
          - 将点云划分为均匀的空间三维体素
          - 优点：可以将卷积池化迁移到3D直接应用
          - 缺点：表达的数据量大，为三次方
      - 典型算法：
          - VoxNet
          - VoxelNet
- Tree-based
      - 基本思想：
          - 使用tree来结构化点云
          - 优点：与体素相比是更高效的点云结构化方法（该粗的粗，细的细）
          - 缺点：让然需要额外的体素化处理
      - 典型算法：
          - OctNet
          - O-CNN
- Point-based
      - 基本思想😕：
          - 直接对点云进行处理，使用对称函数解决点的无序性(e.g., maxpooling)，使用空间变换解决旋转/平移性
      - 典型算法：
          - PointNet(CVPR2017)
              - maxpooling解决无序性
              - 空间变换解决旋转问题：三维的STN(Spatial Tranformation Network)可以通过点云本身的位姿信息学习到一个最有利于网络进行分类或分割的变换矩阵，将点云变换到合适的视角（例如俯视图看车，正视图看人）
          - PointCNN 


#### 1. 车道线检测
👉 https://github.com/andylei77/
1. 基于传统方法的车道线检测
- cv2.Canny
- cv.HoughLinesP(Hough Transform)
2. 基于深度学习(lane-detection summary👉 https://github.com/amusi/awesome-lane-detection)
- LaneNet: 一般将图像透视变换到鸟瞰图，可以进一步优化车道线

  论文：Towards End-to-end Lane Detection: an Instance Segmentation Approach(2018)[https://ieeexplore.ieee.org/abstract/document/8500547?casa_token=8oi_2lJ_OIgAAAAA:i6iIWUnbsRFKrsev6V5HWTCzau090LEdr0AP52crOOtzvJPv12pqrf9fCgKF_h_VDRXdNa3vfLSV


### target tracking(目标跟踪)

## Prediction(sequence data network, behavior modeling)
学术团队：李飞飞(行人), Apollo（车辆）


PNC的重要性逐渐凸显，当前自动驾驶出错50%是PNC模块导致。

要求：
    - Real time 实时
    - Accuracy 准确性

methods:
    - model-based
    - data-driven
    
### Vehicle Predict

Lane Model
    - lane sequence
        - HD map
        - junction
        - off line
    - classification
        - e.g., lane0$\rightarrow$lane1$\rightarrow$lane2
    - lane feature
        - lane S/L
        - lane curvature
        - traffic law
    - vehicle state
        - velocity
        - heading
        - type
        - size
        - heading rate
    - environment
    - network

Data pipeline
- sampling engineering 样本工程
- model optimization
- feature engineering

Trajectory Builder:
拟合目标从A点到B点的运动轨迹
- Kalman filter
- polynomial
- velocity

### Pedestrain Predict
- High randomness
- Low traffic constriants
- No kinematics model
- Benchmark
      - ETH
      - UCY
- SOTA
  - Li Feifei: Social LSTM: Human Trajectory Prediction in Crowded Spaces
 

## (Motion) Planning

### Motion Planning的三个领域
cited from motion planning by Steve Lavelle: http://planning.cs.uiuc.edu/par1.pdf

#### 基础知识
- Robotics Fields:
      - 生成轨迹实现目标
      - RRT, A*, D*, D* Lite
- Control Theory
      - 动态系统理论实现目标状态
      - MPC, LQR
- AI: 生成状态和Action的一个映射
      - Reinforcement Learning, Imitation Learning

Motion planning问题可以简化为一个路径选择问题(最短路径问题)，常见的算法有BFS, DFS, Dijkstra，缺点是均为Non-informative search，效率比较低。经典的A* search为基于Dijkstra的改进算法，知道了终点位置，启发式的。👉[https://www.redblobgames.com/pathfinding/a-star/introduction.html]

自动驾驶的规划和A* search的gap：
    - 部分感知
        - 基于部分感知，自然的想到使用贪心算法：incremental search：目前状态求解到最优
        - D*：部分环境的一个search
            - Apollo登月小车
        - D* Lite
            
    - 动态障碍物
    - 复杂环境
    - A* search本身是一个global algorithm，应用场景为global routing

Autonomous Driving Motion Planning Overall Summary：
    - Safely
    - Smoothly
    - 到达目标地
    - 3D路径优化
    - 硬件
        - 定位感知设备
    - 软件
        - 动态信息
        - 静态信息
            - HD Map
                - 实时性保证（导航信息）


基本planning方法：
- planning路径规划算法可分为四类
  👉[Intro](https://blog.csdn.net/CV_Autobot/article/details/139016301?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%9F%BA%E4%BA%8E%E9%87%87%E6%A0%B7%E5%92%8C%E6%8F%92%E5%80%BC%E7%9A%84%E8%B7%AF%E5%BE%84%E8%A7%84%E5%88%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-139016301.142^v100^pc_search_result_base9&spm=1018.2226.3001.4187)
    - 基于采样: RRT
    - 基于搜索: A*
    - 基于插值拟合: beta spline
    - 基于最优化Numerical Optimization: MPC
- 经典基于环境建模的方法
    - RRT
    - Lattice
- 现代无人车planning的方法
    - Darpa
    - Lattice in Frenet Frame
    - Spiral Polynomial
    👉 A Review of Motion Planning Techniques for Automated Vehicles
- 质点模型
      - 物体看成一个质点
      - 目标为点之间不碰撞
- 刚体问题
      - 车无法简化为质点，为一个刚体
      - BycicleModel
      - XY Heading
      - Collision
- Planning限制条件
      - 避免碰撞
      - 边界阈值
- 连续空间怎么解？
      - 离散化
      - 网格化


传统机器人基础
- PRM(Probabilistic Roadmap Planning)
      - 连续空间离散化
          - 随机撒点
          - 删除障碍物上的点
      - 连接可行点，形成可行空间
      - A*
      - 局限性：为全局算法，通常无法感知全局
- RRT(Incremental version of PRM)
      - 使用增量搜索的方式
      - 找附近可行点的最优解（撒点搜索距离不能太远）
      - 局限性：路径为折线，不平滑
- Lattice
      - 改进了RRT的折线问题，给出了path的平滑曲线
      - 网格化：每个采样格均用曲线连接
      - 局限性：指数级别的一个搜索算法（NP-Hard）
- DP(动态规划)
      - 减小了Lattice的搜索空间，通过复用已有结果
      - 局限性：平滑度仍然不够，曲率连续但曲率导数不一定连续
- QP(二次规划)
      - 凸优化问题最优化求解
      - 通过QP找到平滑曲线
- 刚体模型（对车进行建模）
      - bicycle model😕

👉 [Algorithm tutorial with code](https://github.com/AtsushiSakai/PythonRobotics)
#### 自动驾驶Planning
- 定义：A点到B点，构建一个车辆运动的轨迹，结合HDMap，Localization和Prediction
- 两个层面：导航层面routing；运动轨迹层面planning

- Routing
      - 导航一条A到B的全局路径
      - 输入：地图，当前位置，目的地
      - 输出：路由
      - 搜索：地图数据转化为图网络，其中节点表示道路，边表示道路连接

    - A*
          - 最经典的路径查找算法 [tutorial](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
          - $F(n) = G(n)+H(n)$
              - $F(n)$表示道路routing的总cost
              - $G(n)$表示起始点到候选点的cost
              - $H(n)$表示候选点通过启发函数得到的目标点cost
- Motion Planning
      - planning理解为高精度，低级别的一个search， trajectory planning
      - 轨迹点： XY, Time, Velocity
      - 规划的约束条件
          - collision
          - comfortable
          - 运动学约束
          - 合法

#### APOLLO如何求解规划问题
  基于path-speed decompostion路径速度解耦规划，并采用EM迭代优化。选取最优的路径曲线，并求解当下的最优的ST，再返回优化路径曲线...


#### 机器学习 in PNC
[tutorial video](https://www.youtube.com/watch?v=zR11FLZ-O9M)
- 强化学习
  强化学习是一种机器学习方法，智能体（agent）在环境（environment）中采取一系列行动（action），环境会根据智能体的行动给予奖励（reward）或惩罚。智能体的目标是通过不断学习，找到一种最优策略（policy），使得在长期的交互过程中获得的累积奖励最大化
  - 学习方式
        - 基于价值的学习value-based
        智能体学习一个价值函数（value function），用于估计在每个状态下采取各种行动所能获得的长期奖励。例如，Q - 学习（Q - Learning）是一种典型的基于价值的算法。智能体通过不断更新 Q - 值（Q - value）来学习最优策略，Q - 值表示在某个状态下采取某个行动后的预期累积奖励
            - Q-Learning
                👉[csdn tutorial](https://blog.csdn.net/qq_39429669/article/details/117948150?ops_request_misc=%257B%2522request%255Fid%2522%253A%252262DAD342-F246-4D1D-9AFC-68EF6AD2DDAC%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=62DAD342-F246-4D1D-9AFC-68EF6AD2DDAC&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-117948150-null-null.142^v100^pc_search_result_base9&utm_term=q%20learning&spm=1018.2226.3001.4187)

                👉[flappy_bird using Q-learning implementation](https://github.com/vietnh1009/Flappy-bird-deep-Q-learning-pytorch?tab=readme-ov-file)
    
            - bellman equation
                $Q(s, a)\leftarrow Q(s, a)+\alpha[r+\gamma max_{a'}Q(s', a')-Q(s, a)]$
                where $s$ is state, $a$ is the action, and $r$ is the instant reward. $\alpha$ is the lr between [0, 1], 较小意味着更新越慢，智能体越依赖过去的经验，较大则会更快适应新的信息但可能会导致学习过程不稳定。$\gamma$ is the discount factor between [0, 1]折扣因子，用于权衡近期奖励和远期奖励的重要性，当为1时，更看重长期激励。$s'$为采取动作后的下一状态。其中右侧$[r+\gamma max_{a'}Q(s', a')-Q(s, a)]$为更新项，表示新的估计Q与当前Q的差异。
        - 基于策略的学习policy-based
        直接学习策略函数，通过优化策略来最大化累积奖励。例如，策略梯度（Policy Gradient）方法，它通过计算策略梯度来更新策略参数，使得策略朝着获得更多奖励的方向改进。
- 模仿学习
  模仿学习也称为学徒学习（apprenticeship learning）或学习演示（learning from demonstration），它是一种通过观察专家（expert）的行为来学习策略的方法。智能体试图模仿专家在各种情况下的行为，从而学会执行任务。
  - 学习方式
  - 行为克隆（Behavior Cloning）：
    这是最直接的模仿学习方法。智能体通过直接模仿专家的行动来学习策略。例如，收集专家在一系列状态下采取的行动数据，然后使用监督学习方法（如神经网络）来训练智能体，使得智能体在给定相同状态时能够输出与专家相似的行动。然而，行为克隆可能会受到分布偏移（distribution shift）问题的影响，即智能体在训练过程中看到的状态 行动对和在实际应用中遇到的情况可能不同，导致性能下降。
  - 逆强化学习（Inverse Reinforcement Learning）：
    假设专家的行为是最优的，通过观察专家的行为来推断出专家所遵循的奖励函数，然后利用这个推断出的奖励函数进行强化学习。例如，观察一个熟练的杂技演员的表演，通过分析他的动作来推测出什么样的动作会得到高奖励（如动作的稳定性、美观性等），然后基于这个奖励函数来训练智能体表演杂技。
  - 与强化学习的关系和区别
    - **关系**：模仿学习可以看作是一种特殊的强化学习，它通过利用专家的示范来加速学习过程，或者在奖励信号难以定义的情况下提供一种学习策略的方法。
    - **区别**：强化学习是通过环境给予的奖励信号来学习最优策略，而模仿学习主要依赖于专家的行为示范。强化学习需要在环境中进行大量的探索来发现好的策略，而模仿学习则试图直接复制专家的成功经验。此外，强化学习通常能够处理没有先验知识的情况，而模仿学习需要有专家行为数据作为学习的基础。
  - Alexandre Attia. Global Overview of Imitation Learning


## 一些资料
https://github.com/ProgramTraveler/Road-To-Autonomous-Driving?tab=readme-ov-file

https://www.bilibili.com/video/BV1b94y1F7KU?spm_id_from=333.788.videopod.episodes&vd_source=d7339479a3c0de1eb46f840baa9a3510&p=6
  

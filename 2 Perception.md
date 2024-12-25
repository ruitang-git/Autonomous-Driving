## Perception
### Part1. Segmentation
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
         - Fully Convolutional Networks(FCN)

            神经网络用作语义分割的开山之作。

            （a）提出了全卷积网络（全连接层替换为卷积层）
            （b）使用了反卷积层（从特征图映射为原图大小，上采样的功能）
         - DeepLab
         引入了空洞卷积
         - Pyramid Scene Parsing Network(PSPNet: CVPR 2017)
         核心贡献为Global Pyramid Pooling，将特征图缩放到几个不同的尺寸，使得特征具有更好的全局和多尺度信息
         - Mask R-CNN(BY Kaiming He: ICCV 2017)
         - U-Net(2015)
      
### Part2. Object Detection
### 2D 基于图片👉[code](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
- 算法流程
    1.  位置：先找到所有的ROI(Region of Interest)
         - method1: Sliding Window(会产生很多无效的框，计算复杂)
         - method2: Region Proposal
             - 使用分割segmentation去做
             - 减小了无效的框
             - selective search：over segmentation, merge adjacent boxes according to similarity [reference](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python)
         - method3: CNN
             - anchor box
             - RPN(from Faster-CNN)
    2. 类别：对每一个ROI做分裂获取类别信息
    3. 位置修正
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
              - 更丰富的default box
              - 更灵活的类别预测（把预测类别的机制从空间位置（cell）中解耦，由default box同时预测类别和坐标，有效解决物体重合？？？）
         - YOLO-v3
              - 更好的基础网络
              - 考虑多尺度
    - 基于点云的检测
### 3D 基于激光雷达点云
- 传统vs深度学习
    - 传统：
        - 基于点云的目标检测： 
        
            分割地面 $\rightarrow$ 点云聚类$\rightarrow$特征提取$\rightarrow$分类
        - 地面分割依赖于认为设计的特征和规则，如设置一些阈值，表面法线等
    - 深度学习
        - 非结构化数据
        - 无序性
        - 数据稀疏
      
- Pixel-based
    - 基本思想：
        - 3D $\rightarrow$ 2D, 三维点云在不同角度的相机投影
        - 再借助2D图像领域的深度学习进行分析
    - 典型算法：
        - MVCNN(Multi-View CNN)
        - MV3D: 输入为BV+FV+RGB
        - AVOD: 输入为BEV+RGB
        - SqueezeSeg: 投影到球面
- Voxel-based
    - 基本思想

         将点云划分为均匀的空间三维体素
        - 优点：可以将卷积池化迁移到3D直接应用
        - 缺点：表达的数据量大，为三次方
    - 典型算法：
        - VoxNet
        - VoxelNet
- Tree-based
    - 基本思想：

        使用tree来结构化点云
        - 优点：与体素相比是更高效的点云结构化方法（该粗的粗，细的细）
        - 缺点：让然需要额外的体素化处理
    - 典型算法：
        - OctNet
        - O-CNN
- Point-based
    - 基本思想😕：

        直接对点云进行处理，使用对称函数解决点的无序性(e.g., maxpooling)，使用空间变换解决旋转/平移性
    - 典型算法：
        - PointNet(CVPR2017)
            - maxpooling解决无序性
            - 空间变换解决旋转问题：三维的STN(Spatial Tranformation Network)可以通过点云本身的位姿信息学习到一个最有利于网络进行分类或分割的变换矩阵，将点云变换到合适的视角（例如俯视图看车，正视图看人）
        - PointCNN 

```markdown
**实例之车道线检测**
👉[code](https://github.com/andylei77/) 
1. 基于传统方法的车道线检测
- cv2.Canny
- cv.HoughLinesP(Hough Transform)
2. 基于深度学习
👉[lane-detection summary](https://github.com/amusi/awesome-lane-detection)
- LaneNet: 一般将图像透视变换到鸟瞰图，可以进一步优化车道线
  论文：Towards End-to-end Lane Detection: an Instance Segmentation Approach(2018)
```

### Part3. target tracking
待补充...

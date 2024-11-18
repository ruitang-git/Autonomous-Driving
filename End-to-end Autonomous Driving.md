## End-to-end Autonomous Driving
[1] End-to-end Autonomous Driving: Challenges and Frontiers[link](https://arxiv.org/pdf/2306.16927)
[2] [paper list](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving/blob/main/papers.md#multi-sensor-fusion)

### Motivation
- Pipline disadvantages:
  - the error get accumulated
  - each module has its own objective, the entire system may not be aligned with a unfied target
  - the information may lost between models(since the infomation passed is trivial for human but not hidden values)
- Advantages:
  - can be jointly trained
  - is optimized towards the ultimate goals
  - shared backbones can increase computational efficiency
  - data-driven optimization has the potential to improve the system by simply scaling trianing resources

### Methods
#### Classification
- Imitation Learning
  - behavior cloning(BC)
  - inverse optimal control(inverse reinforcement learning: IRL)
- Reinforcement learning

#### 1.1 Behavior Learning
Utilized an end-to-end neural network to generate control signals from multi-sensor inputs.
- adavantages:
   does not require hand-crafted reward design
- disadvantages:
  - covariate shift
    - 在行为学习（Behavior Learning）中，协变量偏移（Covariate Shift）是一个重要的概念。它指的是在训练数据和测试数据（或者说源数据和目标数据）之间，输入特征（协变量）的分布发生了变化。
    - 例如，在训练一个机器人的行为学习模型时，训练数据是在特定环境（如室内环境，光线充足、地面平坦）下收集的机器人传感器数据（如摄像头图像、激光雷达数据等），这些数据构成了一个输入特征的分布。当机器人被部署到一个新的环境（如室外环境，光线变化大、地面有起伏）时，输入特征的分布就发生了改变，这就是协变量偏移。
  - causal confusion

#### 1.2 Inverse Optimal Control
Traditional IOC algorithms learn an unknown reward function $R(s, a)$ from expert demonstrations.传统的最优控制是给定目标或成本函数后，找到一个最优的控制策略使得系统的性能最优。而在IOC中，目标是给定一些实际的控制行为，推断出可能的成本函数或目标，使得这个推断的陈本函数能解释观察到的行为是最优的。

#### 2.1 Reinforcement Learning

### SOTA Model
- UniAD
- VAD
- UAD
- SparseDrive
- FusionAD
- Hydra-MDP
### Benchmarking

There are 3 approaches for benchmarking end-to-end autonomous driving systems: (1) real-world evaluation, (2) online or closedloop evaluation in simulation, and (3) offline or open-loop
evaluation on driving datasets. 
  

  

## Planning

### Planning的三个领域
cited from motion planning by Steve Lavelle: http://planning.cs.uiuc.edu/par1.pdf

- Robotics Fields:
    - 生成轨迹实现目标
    - RRT, A*, D*, D* Lite
- Control Theory
    - 动态系统理论实现目标状态
    - MPC, LQR
- AI: 生成状态和Action的一个映射
    - Reinforcement Learning, Imitation Learning

Motion planning问题可以简化为一个路径选择问题(最短路径问题)，常见的算法有BFS, DFS, Dijkstra，缺点是均为Non-informative search，效率比较低。经典的A* search为基于Dijkstra的改进算法，知道了终点位置，启发式的。👉 https://www.redblobgames.com/pathfinding/a-star/introduction.html

自动驾驶的规划和A* search的gap：

部分感知
- 基于部分感知，自然的想到使用贪心算法：incremental search：目前状态求解到最优
- D*：部分环境的一个search
    - Apollo登月小车
- D* Lite            
- 动态障碍物
- 复杂环境
- A* search本身是一个global algorithm，应用场景为global routing


### planning分类
planning路径规划算法可分为四类
 👉[Intro](https://blog.csdn.net/CV_Autobot/article/details/139016301?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%9F%BA%E4%BA%8E%E9%87%87%E6%A0%B7%E5%92%8C%E6%8F%92%E5%80%BC%E7%9A%84%E8%B7%AF%E5%BE%84%E8%A7%84%E5%88%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-139016301.142^v100^pc_search_result_base9&spm=1018.2226.3001.4187)
- 基于采样: RRT
- 基于搜索: A*
- 基于插值拟合: beta spline
- 基于最优化Numerical Optimization: MPC


### 现代无人车planning基础知识

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


### 传统机器人基础
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


### 机器学习 in PNC
[tutorial video](https://www.youtube.com/watch?v=zR11FLZ-O9M)
- 强化学习

强化学习是一种机器学习方法，智能体（agent）在环境（environment）中采取一系列行动（action），环境会根据智能体的行动给予奖励（reward）或惩罚。智能体的目标是通过不断学习，找到一种最优策略（policy），使得在长期的交互过程中获得的累积奖励最大化

- 学习方式(强化学习)

    - 基于价值的学习value-based
        智能体学习一个价值函数（value function），用于估计在每个状态下采取各种行动所能获得的长期奖励。例如，Q - 学习（Q - Learning）是一种典型的基于价值的算法。智能体通过不断更新 Q - 值（Q - value）来学习最优策略，Q - 值表示在某个状态下采取某个行动后的预期累积奖励
        - Q-Learning

            👉[csdn tutorial](https://blog.csdn.net/qq_39429669/article/details/117948150?ops_request_misc=%257B%2522request%255Fid%2522%253A%252262DAD342-F246-4D1D-9AFC-68EF6AD2DDAC%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=62DAD342-F246-4D1D-9AFC-68EF6AD2DDAC&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-117948150-null-null.142^v100^pc_search_result_base9&utm_term=q%20learning&spm=1018.2226.3001.4187)

            👉[flappy_bird using Q-learning implementation](https://github.com/vietnh1009/Flappy-bird-deep-Q-learning-pytorch?tab=readme-ov-file)

        - 基于策略的学习policy-based
        直接学习策略函数，通过优化策略来最大化累积奖励。例如，策略梯度（Policy Gradient）方法，它通过计算策略梯度来更新策略参数，使得策略朝着获得更多奖励的方向改进。
- 模仿学习

模仿学习也称为学徒学习（apprenticeship learning）或学习演示（learning from demonstration），它是一种通过观察专家（expert）的行为来学习策略的方法。智能体试图模仿专家在各种情况下的行为，从而学会执行任务。
- 学习方式（模仿学习）
  - 行为克隆（Behavior Cloning）：
    这是最直接的模仿学习方法。智能体通过直接模仿专家的行动来学习策略。例如，收集专家在一系列状态下采取的行动数据，然后使用监督学习方法（如神经网络）来训练智能体，使得智能体在给定相同状态时能够输出与专家相似的行动。然而，行为克隆可能会受到分布偏移（distribution shift）问题的影响，即智能体在训练过程中看到的状态 行动对和在实际应用中遇到的情况可能不同，导致性能下降。
  - 逆强化学习（Inverse Reinforcement Learning）：
    假设专家的行为是最优的，通过观察专家的行为来推断出专家所遵循的奖励函数，然后利用这个推断出的奖励函数进行强化学习。例如，观察一个熟练的杂技演员的表演，通过分析他的动作来推测出什么样的动作会得到高奖励（如动作的稳定性、美观性等），然后基于这个奖励函数来训练智能体表演杂技。
  - 与强化学习的关系和区别
    - **关系**：模仿学习可以看作是一种特殊的强化学习，它通过利用专家的示范来加速学习过程，或者在奖励信号难以定义的情况下提供一种学习策略的方法。
    - **区别**：强化学习是通过环境给予的奖励信号来学习最优策略，而模仿学习主要依赖于专家的行为示范。强化学习需要在环境中进行大量的探索来发现好的策略，而模仿学习则试图直接复制专家的成功经验。此外，强化学习通常能够处理没有先验知识的情况，而模仿学习需要有专家行为数据作为学习的基础。
  - Alexandre Attia. Global Overview of Imitation Learning

> *补充材料之Bellman Equation(Q-Learning)\
> $Q(s, a)\leftarrow Q(s, a)+\alpha[r+\gamma max_{a'}Q(s', a')-Q(s, a)]$\
where $s$ is state, $a$ is the action, and $r$ is the instant reward. \
$\alpha$ is the lr between [0, 1], 较小意味着更新越慢，智能体越依赖过去的经验，较大则会更快适应新的信息但可能会导致学习过程不稳定。\
$\gamma$ is the discount factor between [0, 1]折扣因子，用于权衡近期奖励和远期奖励的重要性，当为1时，更看重长期激励。\
$s'$为采取动作后的下一状态。\
其中右侧$[r+\gamma max_{a'}Q(s', a')-Q(s, a)]$为更新项，表示新的估计Q与当前Q的差异。*

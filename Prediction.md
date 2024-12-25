## Prediction
sequence data network, behavior modeling

学术团队：李飞飞(行人), Apollo（车辆）

    
### Part1. Vehicle Predict
Lane Model
- lane sequence
    - HD map
    - junction
    - off-line
    - classification
        - e.g., lane0 $\rightarrow$ lane1 $\rightarrow$ lane2
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

### Part2. Pedestrain Predict
- High randomness
- Low traffic constriants
- No kinematics model
- Benchmark
      - ETH
      - UCY
- SOTA
  - Li Feifei: Social LSTM: Human Trajectory Prediction in Crowded Spaces

### 节点与节点管理器
- **节点Node** -- 执行单元
    - 执行具体任务的进程、独立运行的可执行文件
    - 不同的节点可使用不同的编程语言，可分布式运行在不同的主机
    - 节点在系统中的名称必须是唯一的
- **节点管理器ROS Master** -- 控制中心
    - 为节点提供命名和注册服务
    - 跟踪和记录话题/服务通信，辅助节点相互查找、建立连接
    - 提供参数服务器，节点使用此服务器存储和检索运行时的参数

 ### 话题通信（异步通信机制，订阅者无反馈，单向）
 重复持续的进行通信
 - **话题Topic** -- 异步通信机制
     - 节点间用来传输数据的重要总线
     - 使用发布／订阅模型，数据由发布者传输到订阅者，同一个话题的订阅者或发布者可以不唯一
- **消息Message** -- 话题数据
    - 具有一定的类型和数据结构，包括ROS提供的标准类型和用户自定义类型
    - 使用编程语言无关的.msg文件定义，编译过程中生成对应的代码文件


### 服务通信Service（同步通信机制，客户端进行反馈，双向）
单次的请求
- 使用客户端/服务端（C/S）模型，客户端发送请求数据，服务端完成处理后返回应答数据
- 使用编程语言无关的.srv文件定义请求和应答数据结构，编译过程中生成对应的代码文件

### 话题与服务通信的区别
|  |话题|服务|
|--|--  |-- |
|同步性|异步|同步|
|通信模型|发布/订阅|服务器\客户端|
|底层协议|ROSTCP/ROSUDP|ROSTCP/ROSUDP|
|反馈机制|无|有|
|缓冲区|有|无|
|实时性|弱|强|
|节点关系|多对多|一对多（一个server）|
|使用场景|数据传输|逻辑处理e.g.,配置参数|

### 参数Parameter -- 全局共享字典
    - 可通过网络访问的共享、多变量字典
    - 节点使用此服务器来存储和检索运行时的参数
    - 适合存储静态、非二进制的配置参数，不适合存储动态配置的数据

### 文件系统
- 功能包（Package）
  ROS软件中的基本单元，包含节点源码、配置文件、数据定义等
- 功能包清单（Package manifest）
  记录功能包的基本信息，包含作者信息、许可信息、依赖选项、编译标志等
- 元功能包（Meta Package）
  组织多个用于同一目的的功能包


### 创建工作空间和功能包

#### 1. 介绍
工作空间workspace是一个存放工程开发相关文件的文件夹

- src: 代码空间
- build：编译空间
- install：安装空间
- devel：开发空间（ROS1中有，ROS2中删去，因为与install空间有重复）
  
```
+----------------------+
+ workspace_folder/    +
+     src/             +
+     |                +
+     build/           +
+     |                +
+     devel/           +
+     |                +
+     install/         +
+     ...              +
+----------------------+
```

#### 2. 创建
- 创建工作空间(创建src目录)
```
$ mkdir ~/catkin_ws/src
$ cd ~/catkin_ws/src
$ catkin_init_wrokspace
```
- 编译工作空间(创建build, devel, install目录)
```
$ cd ~/catkin_ws
$ catkin_make
$ catkin_make install
```
- 设置环境变量
```
$ source ~/catkin_ws/devel/setup.bash
```
- 检查环境变量
```
$ echo $ROS_PACKAGE_PATH
```

#### 3. 创建功能包
功能包为代码的最小单元，代码不能直接放在src目录下！
**catkin_create_pkg <package_name> [depend1][depend2][depend3]**
- 创建功能包
```
$ cd ~/catkin_ws/src
$ catkin_create_pkg test_pkg rospy roscpp
```
- 编译功能包
- ```
$ cd ~/catkin_ws
$ catkin_make
$ source ~/catkin_ws/devel/setup.bash
```

同一个工作空间下，不允许存在同名功能包。不同空座空间下，允许存在同名功能包

### 通信

#### 1. 发布者Publisher
如何实现一个发布者？
step1. 创建发布者代码（c++/python）
- 初始化ROS节点
- 向ROS Master注册节点信息，包括发布的话题名和消息类型
- 创建消息数据
- 按照一定频率循环发布消息
step2. 配置发布者代码编译规则
如何配置CMakeLists.txt中的编译规则
- 设置需要编译的代码和生成的可执行文件
```
add_executable(velocity_pyblisher src/velocity_publisher.cpp)
```
- 设置链接库
```
target_link_libraries(velocity_publisher ${catkin_LIBRARIES})
```
step3. 编译并运行
```
## 编译
$ cd ~/catkin_ws
$ catkin_make   ## 编译
$ source devel/setup.bash ## 编译结束后一定要记得配置环境变量！！！
## 运行
$ roscore
$ rosrun turtlesim turtlesim_node
$ rosrun learning_topic velocity_publisher
```

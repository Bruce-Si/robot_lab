# 无人机导航与抓取交接文档

本文档说明 `robotlab_navigrasp` 中无人机导航避障、抓取与放置的训练和评估流程。

## 运行环境

使用 Isaac Lab 2.3.2 和 Isaac Sim 5.1：

```bash
cd path/to/robotlab_navigrasp
source path/to/conda/etc/profile.d/conda.sh
conda activate isaaclab_v2.3.2
```

在已安装并激活 Isaac Lab 的环境中，下面的脚本均直接使用 `python` 启动。
请从 `robotlab_navigrasp` 根目录执行命令。


除非特别说明，默认使用 `--device cuda:0`。不需要 GUI 时添加 `--headless`。

## 已注册任务和场景

| 任务 ID | 场景/配置 | 用途 |
| --- | --- | --- |
| `RobotLab-Navigation-Tilting-UAV-v0` | 8x8 个 cell 组成的柱子 USD 网格 | 导航训练和评估 |
| `RobotLab-Navigation-Tilting-UAV-Grasp-v0` | `Cell_7_7` 加桌子、立方体和放置标记 | 导航后的完整拾取与放置 |

8x8 场景使用 36 束水平 LiDAR，角分辨率为 10 度，没有竖直方向线束。

导航交接条件：

```text
到目标点的相对距离 < 1.0 m
相对 yaw 绝对值 < 10 度
```

**组合抓取任务中的桌子和立方体仍然具有物理碰撞，但它们位于 LiDAR 目标根节点之外。因此导航 LiDAR 不会把桌子和待抓物当成导航障碍物。**

## 物理频率和策略频率

默认时序：

```text
物理积分频率：400 Hz，dt = 0.0025 s
策略/环境步频：10 Hz，0.1 s
decimation：每个策略步包含 40 个物理步
LiDAR 更新频率：10 Hz
```

`--physics-hz` 只修改物理积分频率，不修改策略步频。因为策略固定为 10 Hz，所以物理频率必须是 10 的整数倍，例如 `400、200、100、60、50`；`75` 会被拒绝。

训练和最终精度验证使用 400 Hz。GUI 测试卡顿时可以使用 60 Hz。使用 60 Hz 评估 400 Hz 训练的模型时，物理积分动力学发生了变化，因此低频结果适合调试和快速测试，不应作为最终精度指标。

频率配置函数位于：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/uav/navigation_env_cfg.py
```

## 强化学习导航策略

当前导航策略使用 RSL-RL 的 PPO 实现。它的定位是**仿真数据采集专家**：优先提高仿真中的导航成功率，为后续 VLA/WAM 数据集提供成功轨迹；它不是直接部署到真机的策略。

| 项目 | 当前配置 |
| --- | --- |
| 算法 | PPO |
| Actor/Critic 网络 | MLP，隐藏层 `[256, 256, 128]`，ELU 激活 |
| 动作分布 | 高斯分布，初始标准差 `0.3`，动作裁剪到 `[-1, 1]` |
| Rollout 长度 | 每个环境 32 个策略步 |
| PPO 更新 | 5 个 epoch，4 个 mini-batch，clip `0.2` |
| 优化器 | 自适应学习率，初始 `1e-3`，目标 KL `0.01`，最大梯度范数 `1.0` |
| 折扣参数 | `gamma=0.995`，`lambda=0.95`，熵系数 `0.003` |
| 观测归一化 | 不使用 RSL-RL 运行时归一化；各输入在观测函数内按固定量纲缩放 |

训练配置位于：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/uav/agents/rsl_rl_ppo_cfg.py
```

### 观测空间

训练时 actor 和 critic **均**接收 `policy + privileged` 两组观测，因此总维度为 `43 + 95 = 138`。这是一种有意使用上帝视角信息的专家策略设计。

`policy` 为 43 维、理论上可由真实传感器获得的输入：

| 内容 | 维度 | 处理方式 |
| --- | ---: | --- |
| 目标相对位置 `(goal_x, goal_y)` | 2 | 机体系目标坐标除以 50 m |
| 机体系线速度 `(v_x, v_y)` | 2 | 除以 1.0 m/s |
| 目标方位 | 2 | `sin(bearing), cos(bearing)` |
| 机体系 yaw 角速度 | 1 | 除以 1.0 rad/s |
| 水平 LiDAR | 36 | 360 度、10 度分辨率、最大 4 m；值为 `1 - distance / 4`，前向线束被循环移到第 0 维 |

`privileged` 为 95 维、只能从仿真器直接得到的输入：本 cell 内无人机与目标精确坐标、定高误差、真实航向、完整三轴线速度和角速度、机体系重力方向、控制器 yaw 跟踪误差、精确 cell 行列坐标与 64 维 cell one-hot、速度积分器状态、4 个电机实际推力和 4 个舵机实际角度。它不适用于真机部署，但适合稳定地批量采集导航数据。

注意：虽然 `policy` 观测本身是部署可见的，当前 actor 仍拼接了 `privileged` 观测。因此若要训练可上真机的策略，必须另建一个只使用 `policy` 的 actor 配置并重新训练，不能直接删除运行时输入。

观测实现位于：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/mdp/uav_navigation.py
```

### 动作空间和底层控制

PPO 每个策略步输出 3 维归一化动作：

```text
[a_vx, a_vy, a_yaw]，每一维范围 [-1, 1]
```

| 动作 | 物理含义 | 当前范围 |
| --- | --- | --- |
| `a_vx` | 机体系前后目标速度 | `[-1, 1]` m/s |
| `a_vy` | 机体系左右目标速度 | `[-1, 1]` m/s |
| `a_yaw` | 目标 yaw 角速度 | `[-1, 1]` rad/s |

训练阶段不由 RL 控制高度、roll、pitch 或夹爪。控制器固定定高在 reset 高度，定高增益为 `1.5`、最大垂直速度为 `0.8 m/s`；roll/pitch 保持为零，夹爪维持打开。平面速度和 yaw 命令经过时间常数 `0.15 s` 的一阶滤波后，由速度 PID、姿态 PID、四倾转旋翼分配器转换为机体 wrench、四路推力和四路舵机角度。

因此，导航模型输出可以直接进入无人机的低层控制器，但只覆盖平面导航。在完整抓取流程中，导航完成后由 `run_tilting_uav_nav_grasp.py` 的脚本阶段接管高度、俯仰和夹爪控制。

动作控制实现位于：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/mdp/uav_velocity_action.py
```

### 奖励函数

所有奖励以 10 Hz 策略频率计算。当前奖励项如下：

| 奖励项 | 权重 | 计算方式 |
| --- | ---: | --- |
| 终止惩罚 | `-200` | 非超时终止时施加 |
| 朝向目标速度 | `+0.5` | 世界系水平速度在目标方向上的投影，裁剪至 `[-1, 1]` |
| 到目标的距离进展 | `+0.5` | 本策略步相对上一步的距离减少量，裁剪至 `[-1, 1]` |
| LiDAR 障碍惩罚 | `+2.0` | 取最近 3 束的平均距离；`d < 1.8 m` 开始线性负奖励，`d <= 0.8 m` 额外加入指数惩罚 `-exp(5*(0.8-d))` |
| 到达目标奖励 | `+200` | 同时满足距离 `< 1.0 m` 且相对 yaw `< 10` 度 |
| 近目标 yaw 惩罚 | `-2.0` | `exp(-(distance/2)^2) * abs(yaw_error)/pi`，只在接近目标时显著 |
| 动作变化惩罚 | `-0.01` | 相邻策略动作差的平方和 |

奖励实现在：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/uav/navigation_env_cfg.py
source/robot_lab/robot_lab/tasks/manager_based/navigation/mdp/rewards.py
```

### 成功、失败和场景采样

每个 episode 最长 100 s。成功条件为同时满足目标平面距离 `< 1.0 m` 和无人机机首相对目标方向的 yaw 误差 `< 10` 度。达到成功条件后 episode 结束，并获得到达目标奖励。

以下情况会提前结束 episode：任一无人机或夹爪刚体接触力超过 `5 N`；最近 2 束 LiDAR 的平均距离小于 `0.45 m`；与定高目标相差超过 `0.5 m`；roll/pitch 倾角超过约 `28.6` 度；或离开所属 50 m cell 的边界 `0.5 m` 缓冲区。

8x8 场景没有启用课程学习。每个 cell 尺寸为 50 m，环境按 cell 循环分配；起点和目标从 cell 的相对两侧边缘采样，边缘偏移为 22 m，横向随机范围为 `[-18, 18]` m。固定为 `Cell_7_7` 的评估和组合抓取脚本不改变训练时的策略定义。

## 导航避障训练

8x8 柱子场景训练，默认 400 Hz：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Navigation-Tilting-UAV-v0 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --physics-hz 400 \
  --device cuda:0
```

常用训练参数：

```text
--num_envs N                 并行环境数量
--max_iterations N           PPO 训练迭代次数
--physics-hz HZ               物理频率，默认 400
--video                       录制训练视频
--video_length N              视频长度，单位为环境步
--video_interval N            录制间隔，单位为环境步
--seed N                      随机种子
--resume                     通过 RSL-RL 参数恢复训练
--eval_interval N             每 N 次迭代进行一次确定性评估
--eval_level ROW              8x8 场景评估行，默认 7
--eval_variant COL            8x8 场景评估列，默认 7
--eval_episodes_per_env N     每个评估环境完成的 episode 数量
```

训练期间的周期评估目前只支持注册的 UAV 导航任务和单进程训练。评估 JSON 会写入当前训练 run 的 `evaluations/` 目录。

## 下载已训练导航模型

当前推荐的导航模型为 GitHub Release `uav-navrl-model-2100` 中的 `model_2100.pt`，对应源码
commit `797a8c4`。从项目根目录下载到组合脚本和本文评估命令使用的默认位置：

```bash
mkdir -p logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle/2026-07-27_21-27-40

curl -L --fail \
  https://github.com/Bruce-Si/robot_lab/releases/download/uav-navrl-model-2100/model_2100.pt \
  -o logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle/2026-07-27_21-27-40/model_2100.pt
```

下载后校验文件完整性：

```bash
sha256sum logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle/2026-07-27_21-27-40/model_2100.pt
```

期望结果：

```text
837c5aa2a8f9677eb18122eac76966225a1cf72f14f437afc5abb246f86a3b6b
```

该模型以 400 Hz 物理频率训练，actor 输入为 138 维（43 维 policy 加 95 维 privileged），
动作为 3 维平面速度和 yaw 角速度。Release 页面：
<https://github.com/Bruce-Si/robot_lab/releases/tag/uav-navrl-model-2100>。

## 导航避障评估

使用 1024 架无人机、评估最难的 `Cell_7_7`，并将物理频率降到 60 Hz：

```bash
python scripts/reinforcement_learning/rsl_rl/eval_uav_navigation.py \
  --task RobotLab-Navigation-Tilting-UAV-v0 \
  --checkpoint logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle/2026-07-27_21-27-40/model_2100.pt \
  --num_envs 1024 \
  --terrain_level 7 \
  --terrain_variant 7 \
  --episodes_per_env 1 \
  --physics-hz 60 \
  --headless \
  --device cuda:0
```

默认评估器会加载一个提取出的 cell，而不是完整 8x8 USD。只有明确需要完整网格时才使用 `--full_grid_scene`。

评估参数：

```text
--checkpoint PATH             RSL-RL 模型路径，必需
--num_envs N                  并行评估无人机数量，默认 1024
--terrain_level ROW           8x8 场景行，默认 7
--terrain_variant COL         8x8 场景列，默认 7
--episodes_per_env N          每架无人机完成的 episode 数量
--physics-hz HZ               物理频率，默认 400
--full_grid_scene             不使用提取出的单 cell，加载完整网格
--output PATH                 显式指定评估 JSON 路径
--seed N                      可复现的 reset/目标随机种子
```

终端摘要会输出成功率、碰撞率、超时率、距离成功统计和 yaw 成功统计；相同数据也会写入 JSON。

## 导航环境 smoke test

不加载模型，快速检查环境注册、传感器和控制接口：

```bash
python scripts/tools/smoke_tilting_uav_navigation_env.py \
  --scene-profile grid_8x8 \
  --num-envs 8 \
  --physics-hz 60 \
  --headless \
  --device cuda:0
```

smoke test 会检查观测和动作维度、水平 LiDAR、定高、yaw 响应、夹爪保持、局部 reset，以及 1 m/10 度导航成功边界。

## 完整导航与拾取放置

当前组合脚本使用最好的导航 checkpoint，然后执行完整的物理拾取与放置流程：

RobotLab 使用的派生 Articulation 层位于：

```text
source/robot_lab/robot_lab/assets/uav_nav.usda
```

该文件只引用 `TiltingUAV_Diffusion` 中的原始 `uav.usd`，原始抓取工程保持只读。夹爪驱动参数、阶段节奏和闭合前停留时间与原始 `GripperPhysics`/脚本专家保持一致；抓取偏移已转换到 RobotLab 的 `/base_link` 坐标系并经过校准。

RobotLab 当前使用的 `/base_link` 抓取根部偏移为 `(-0.22, 0, +0.13) m`。不要直接把原始顶层 UAV 位姿中的 `(-0.30, 0, +0.05) m` 当作 RobotLab 根部偏移；两者的参考坐标系和俯仰姿态不同。

抓取任务将闭合目标设为 `(left=+0.020, right=-0.020) m`，对应当前 `0.04 m` 宽立方体。两个关节都驱到 `0` 会让右指穿过立方体并只产生侧向推移；更换物体尺寸时应按物体宽度重新标定该目标。

```text
NAVIGATE -> APPROACH -> DESCEND -> PRE_CLOSE -> CLOSE -> LIFT -> CARRY -> PLACE_DESCEND
-> PRE_RELEASE -> RELEASE -> RECOVER -> DONE
```

流程说明：

1. 导航到桌子前，要求距离小于 1 m 且相对 yaw 小于 10 度。
2. 下降到抓取位姿，保持夹爪张开停留 1 s，等待 UAV 和立方体稳定。
3. 闭合夹爪并保持 1 s。
4. 抬升立方体超过 0.10 m。
5. 保持夹爪闭合，水平搬运到抓取桌前方地面上的绿色目标区。
6. 下降到放置高度，闭爪停留后打开夹爪。
7. 检查立方体位置、z 高度和线速度，确认放置成功。
8. 无人机抬升回撤，夹爪保持打开，进入 `DONE`。

放置点目前固定在：

```text
object target = (398.30, 375.00, 0.04) m
```

运行完整 GUI 流程，使用 60 Hz 降低卡顿：

```bash
python scripts/tools/run_tilting_uav_nav_grasp.py \
  --checkpoint logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle/2026-07-27_21-27-40/model_2100.pt \
  --physics-hz 60 \
  --duration 180 \
  --device cuda:0
```

组合脚本参数：

```text
--checkpoint PATH              导航 checkpoint
--duration SEC                 最大仿真时长，默认 180 s
--hold-seconds SEC             放置回撤后的最终保持时间，默认 5 s
--pre-release-seconds SEC      放置位置闭爪停留时间，默认 1 s
--release-hold-seconds SEC     打开夹爪后的物体稳定时间，默认 1 s
--pre-close-seconds SEC        抓取位姿张开夹爪停留时间，默认 1 s
--grasp-hold-seconds SEC       闭合夹爪后抬升前停留时间，默认 1 s
--status-hz HZ                 终端状态输出频率，默认 1 Hz
--physics-hz HZ                物理频率，默认 400 Hz
--navigation-only              只运行到导航交接，不执行抓取放置
--show-lidar                   GUI 中显示水平 LiDAR 线束
```

不使用 `--show-lidar` 时，LiDAR 仍然正常参与观测和逻辑，只关闭线束 debug overlay。机载 `Camera_OmniVision_OV9782_Color` 会在 GUI 中被设置为 active camera。

## 旧项目中的抓取流程

旧的独立 embedded demo：

```text
TiltingUAV_Isaac/apps/embedded/run_isaac_embedded_grasp_demo.py
```

它的流程也是：

```text
TAKEOFF -> APPROACH -> DESCEND -> GRASP -> LIFT -> HOLD
```

它只验证抬升，不会放置立方体。

旧的 Diffusion scripted expert 位于：

```text
TiltingUAV_Isaac/src/diffusion/scripted_expert_planner.py
```

它包含完整的后续阶段：

```text
LIFT -> XY_PLACE -> PLACE_DESCEND -> PRE_RELEASE -> RELEASE -> RECOVER -> DONE
```

当前 RobotLab 组合脚本已经实现相同的高层搬运和释放逻辑，并使用 Cell_7_7 的桌前地面目标点。

## 常见问题

GUI 卡顿时，先使用 `--physics-hz 60` 和单环境测试。测吞吐量时使用 `--headless`。训练和最终精度验证保持 400 Hz。

如果组合抓取任务再次出现 `PxDirectGPUAPI::copyContactData()`，检查 `scene.contact_forces` 是否仍为 `None`。这只关闭未使用的 GPU ContactSensor 数据流，不应删除桌子、立方体或无人机的物理碰撞。

静态场景事件只验证 USD 中已经写入的 `CollisionAPI`，不会运行时补写。如果 grid smoke test 报告 `/World/uav_navigation_scene/GlobalGround` 缺少 `CollisionAPI`，说明当前完整网格 USD 的地面碰撞标注需要修复。这个问题与 `--physics-hz` 无关。

如果运行在指定时长结束，表示脚本达到时间上限，不一定表示导航失败。完整拾取与放置流程需要确认日志依次出现 `CARRY`、`RELEASE`、`place_ok=True`、`DONE`，以及最后的 `Pick -> Place: PASS`。

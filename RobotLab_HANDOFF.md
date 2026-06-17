# 四足 Navigation + Locomotion 项目交接文档

更新时间：2026-06-15

当前代码分支：

```text
feature/go2-navrl-navigation
```

当前 fork：

```text
https://github.com/Bruce-Si/robot_lab/tree/feature/go2-navrl-navigation
```

## 1. 项目目标

本项目是在 IsaacSim/IsaacLab/RobotLab 中实现一套四足机器人分层强化学习控制算法，用于完成 Navigation + Locomotion 任务。

整体采用上下层策略结构：

- 上层导航策略根据 LiDAR 深度图、目标相对状态和机器人运动状态，输出导航避障速度指令。
- 下层步态策略接收上层速度指令和本体感知，输出关节位置，驱动四足机器人行走。

当前实验对象是 Unitree Go2。下层 locomotion policy 已经训练好并冻结；当前主要训练和调参对象是上层 navigation policy。

当前任务注册名：

```bash
RobotLab-Navigation-Go2-v0
```

任务注册位置：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/go2/__init__.py
```

## 2. 代码结构

导航任务主要代码位于：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/
```

核心文件：

```text
config/go2/navigation_env_cfg.py
config/go2/agents/rsl_rl_ppo_cfg.py
mdp/observations.py
mdp/rewards.py
mdp/terminations.py
mdp/curriculums.py
mdp/commands.py
mdp/heading_locked_action.py
```

调试工具：

```text
scripts/tools/lidar_viewer.py
```

USD 场景生成/处理工具：

```text
scripts/tools/generate_navrl_usd.py
scripts/tools/generate_usd_in_sim.py
scripts/tools/bake_usd_cones_to_mesh.py
scripts/tools/convert_usda_to_usd.py
scripts/tools/parse_usd_obstacles.py
scripts/tools/inspect_usd_prim.py
```

训练脚本有一处扩展：

```text
scripts/reinforcement_learning/rsl_rl/train.py
```

新增参数：

```bash
--load_scene
```

该参数会设置：

```text
NAVRL_USD_SCENE
NAVRL_OBSTACLE_JSON
```

用于加载外部 USD 障碍物场景。

## 3. 场景与环境

默认 procedural terrain 配置在：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/go2/navigation_env_cfg.py
```

主要参数：

- terrain grid：8 x 8，共 64 个 unique obstacle layouts
- cell size：50m x 50m
- obstacle 数量：从 15 逐步增加到 90
- obstacle 高度：2.25m
- obstacle 尺寸：从 0.1m 到 2.0m
- 初始最大 terrain level：`max_init_terrain_level=3`
- terrain curriculum：开启

如果通过 `--load_scene` 使用 USD 场景，配置会切换为：

- 不再生成 procedural terrain
- 加载指定 USD 作为唯一场景几何
- 使用 `MultiMeshRayCasterCfg` 对 USD 场景做 raycast
- 为 curriculum 创建 synthetic terrain-level state

当前边界策略：

- `terrain_out_of_bounds = None`
- 不对越界做 reward 或 termination 惩罚
- 边界主要依赖场景中的物理围墙限制

## 4. 分层控制与动作空间

当前上层 action 使用 heading-locked 封装：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/mdp/heading_locked_action.py
```

上层 policy 输出 2 维动作：

```text
[vx_body, yaw_rate]
```

发送给下层 locomotion policy 的速度命令为：

```text
[vx_body, 0.0, yaw_rate]
```

也就是说当前锁定 `vy=0`。机器人不能直接横向移动，想向左或向右走，需要先转动机头再前进。

做这个实验的原因：

- 之前允许 `vy` 时，轨迹中出现过类似横移绕圈、螃蟹步接近目标的行为。
- reward 同时鼓励朝向目标，可能和横向移动形成冲突。
- 当前先用 `vy=0` 验证是否能得到更稳定、更自然的导航行为。

debug 箭头：

- 绿色箭头：实际发送给下层的线速度命令 `[vx, vy]`
- 蓝色箭头：机器人当前 body-frame 线速度
- 已修复此前把 `yaw_rate` 误画成横向速度的问题

## 5. 观测输入

PPO actor/critic 使用两个 observation group：

```python
obs_groups = {
    "actor": ["policy", "lidar"],
    "critic": ["policy", "lidar"],
}
```

### 5.1 低维 state

函数：

```text
mdp/observations.py::navrl_state
```

当前 state 为 8 维：

```text
[goal_pos_body / 50.0, root_lin_vel_b, sin(goal_heading_b), cos(goal_heading_b)]
```

含义：

- `goal_pos_body / 50.0`：目标在机器人 body frame 下的相对位置，3 维，使用固定 50m 尺度归一化
- `root_lin_vel_b`：机器人 base 线速度，body frame，3 维
- `sin(goal_heading_b), cos(goal_heading_b)`：目标方位角，2 维

之前曾使用“episode 初始化时起点到终点的距离”做 goal 归一化，后来改为固定 50.0m。固定尺度更稳定，不会因为不同 episode 起终点距离不同导致 state 尺度变化。

### 5.2 LiDAR

配置：

```text
72 horizontal beams x 8 vertical beams
range = 4.0m
vertical_fov_range = (-10 deg, 20 deg)
horizontal_fov_range = (-180 deg, 180 deg)
LiDAR offset = (0.0, 0.0, 0.3)
```

函数：

```text
mdp/observations.py::lidar_depth
```

策略输入已改为 NavRL 风格 proximity：

```text
proximity = 1.0 - distance / max_distance
```

含义：

- 远处或无障碍：接近 0
- 近处障碍：接近 1

注意：

- policy 输入是 proximity
- `/tmp/navrl_lidar.npy` 保存的是 raw distance，单位 m，用于 viewer 显示
- viewer 里看到的是距离图，不是 proximity 图

### 5.3 Observation normalization

当前 RSL-RL CNN model 配置：

```python
obs_normalization = False
```

也就是没有使用 RSL-RL 内置 observation normalization。当前依赖手动处理尺度：

- goal 相对位置除以 50.0
- LiDAR 转成 `[0, 1]` proximity
- heading 用 sin/cos 表示
- 速度目前未额外归一化

## 6. 网络结构与 PPO

配置文件：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/go2/agents/rsl_rl_ppo_cfg.py
```

当前 actor/critic：

- `RslRlCNNModelCfg`
- CNN 编码 LiDAR
- MLP 融合低维 state
- actor/critic 共享 CNN encoder
- activation：`elu`
- hidden dims：`[256, 256]`

CNN 配置：

```python
output_channels=[4, 16, 32, 32]
kernel_size=[5, 5, 3, 3]
stride=[1, 2, 2, 2]
padding="zeros"
activation="elu"
flatten=True
```

当前 PPO 参数：

```python
num_steps_per_env = 24
max_iterations = 5000
clip_actions = 1.0
init_std = 1.0
std_type = "scalar"
entropy_coef = 0.003
gamma = 0.995
lam = 0.95
learning_rate = 1e-3
schedule = "adaptive"
desired_kl = 0.01
num_learning_epochs = 5
num_mini_batches = 4
max_grad_norm = 1.0
```

`mean_std` 观察经验：

- 当前 2D clipped action 下，比较健康的最终范围大致是 0.3 到 0.6
- 低于 0.15 通常表示探索过早收缩
- 高于 2 通常可疑
- 之前 `entropy_coef=0.01` 时出现过 `mean_std` 爆炸到 38+ 的情况

## 7. Reward

配置位置：

```text
navigation_env_cfg.py::RewardsCfg
```

当前 reward：

```python
termination_penalty: weight = -200.0
vel_towards_goal: weight = 0.5
progress_towards_goal: weight = 5.0
lidar_obstacle: weight = 2.0
goal_bonus: weight = 200.0
action_rate_l2: weight = -0.01
```

当前是 dense + sparse reward：

- dense：朝目标速度、距离进展、LiDAR 避障、动作平滑
- sparse：到达目标 `goal_bonus=200`
- failure termination：`termination_penalty=-200`

`goal_reached` 被设置为 `time_out=True`，所以成功到达目标不会吃 `termination_penalty`。

### 7.1 LiDAR obstacle penalty

函数：

```text
mdp/rewards.py::lidar_obstacle_penalty
```

当前参数：

```python
k_nearest = 5
soft_threshold = 1.8
hard_threshold = 0.8
max_distance = 4.0
```

当前逻辑：

```python
nearest_k = top-k nearest LiDAR distances
d = mean(nearest_k)
soft = clamp(d - soft_threshold, min=hard_threshold - soft_threshold, max=0.0)
hard = -exp(5.0 * (hard_threshold - d)) if d <= hard_threshold else 0
return soft + hard
```

这个函数返回负值或 0。配置中的 reward weight 是正数 2.0，因此最终加入总 reward 后仍然是惩罚。

历史上发现过 soft penalty clamp 写反的问题。如果 clamp 方向写错，会导致距离越近惩罚反而越弱，或者符号不符合预期。当前修正后的关键代码是：

```python
soft = torch.clamp(d - soft_threshold, min=hard_threshold - soft_threshold, max=0.0)
```

### 7.2 progress reward 注意事项

`progress_towards_goal` 会更新环境内部状态：

```python
env._nav_prev_goal_distance
```

因此 debug 代码不要重新调用 reward function，否则会改变训练状态。viewer 读取的是 `reward_manager._step_reward`，不会重复调用 reward 函数。

## 8. Termination

配置位置：

```text
navigation_env_cfg.py::TerminationsCfg
```

当前 termination：

- `time_out`
- `lidar_collision`
- `fallen_over`
- `goal_reached`

已禁用：

```python
terrain_out_of_bounds = None
```

`lidar_collision` 判定：

```text
mean(k nearest lidar distances) < body_radius
body_radius = 0.4
k_nearest = 5
```

`fallen_over` 判定：

- reset 后 50 steps grace period
- 连续 40 steps 几乎不移动
- body tilt 超过约 75 度

注意：`navigation_env_cfg.py` 里有一段旧注释说 collision 不结束 episode，但当前代码中 `lidar_collision` 实际是 termination，会结束 episode。后续如果修改 collision 逻辑，需要同步修正文档和注释。

## 9. Curriculum

配置位置：

```text
navigation_env_cfg.py::CurriculumCfg
```

当前设置：

```python
success_threshold = 8
failure_threshold = 4
```

逻辑：

- 连续成功达到阈值，terrain level + 1
- 连续失败达到阈值，terrain level - 1
- 成功判定：目标距离 `< 1.0m`

放慢 curriculum 是目前最明确有效的改动之一。单独放慢 curriculum 后，成功率峰值曾提升到 0.8+。

## 10. Debug Viewer

viewer 文件：

```text
scripts/tools/lidar_viewer.py
```

环境侧写出：

```text
/tmp/navrl_lidar.npy
/tmp/navrl_state.npy
/tmp/navrl_levels.npy
/tmp/navrl_rewards.npz
```

viewer 当前显示：

- 左侧：reward 各项曲线
- 左下：reward label 和最新值
- 右上：env 状态、目标距离、heading error、速度、terrain level 分布
- 右下：LiDAR 距离图

reward 曲线：

- 只保留最近 100 个点
- 横坐标是当前选中 env 的 episode step
- env reset 后自动清空上一段历史，避免把不同 episode 连起来
- reward 数据来自 `reward_manager._step_reward`

启动：

```bash
python scripts/tools/lidar_viewer.py
```

操作：

- 左右方向键：切换 env
- 输入数字后 Enter：跳转指定 env

注意：如果训练进程已经在跑，修改 `observations.py` 后不会被当前训练进程自动加载，需要重启训练/仿真进程。

## 11. 最近训练实验结论

日志目录：

```text
logs/rsl_rl/unitree_go2_navrl/
```

关键 run：

```text
2026-06-13_09-01-30_vy0_std1.0_entropy0.01
2026-06-14_17-35-16_vy0_std0.5_entropy0.001
2026-06-15_09-12-18_vy0_std1.0_entropy0.001
2026-06-15_14-26-08
```

主要结论：

1. `vy=0 + slow curriculum + init_std=1.0 + entropy=0.01`
   - 成功率峰值约 0.84
   - 但 `mean_std` 爆炸到 38+，探索过强，训练不稳定

2. `vy=0 + init_std=0.5 + entropy=0.001`
   - 效果明显变差
   - `mean_std` 下降到约 0.1
   - 成功率峰值约 0.58
   - 探索过早收缩

3. `vy=0 + init_std=1.0 + entropy=0.001`
   - 早期好于低 std 版本
   - 成功率曾到约 0.78
   - 但 std 仍持续下降，后期不够稳定

4. 当前折中配置
   - `init_std=1.0`
   - `entropy_coef=0.003`
   - 目标是在避免 std 爆炸和避免 std collapse 之间取中间值

当前最值得继续完整跑完并观察的是第 4 类配置，也就是现在代码里的 PPO 配置。

## 12. 重要历史改动记录

按大致顺序：

1. 修正 LiDAR obstacle soft penalty 的符号和 clamp 方向
2. 移除靠近 terrain 边界的惩罚
3. 禁用 `terrain_out_of_bounds`
4. goal 相对位置归一化改成固定 50.0m
5. LiDAR policy 输入改为 `1 - distance / max_distance`
6. 高层 action 从 `[vx, vy, yaw_rate]` 改为 `[vx, yaw_rate]`，锁定 `vy=0`
7. 修复 viewer/debug 速度箭头显示
8. 放慢 curriculum：`success_threshold=8, failure_threshold=4`
9. PPO std/entropy 实验：
   - `init_std=0.5, entropy=0.001` 过保守
   - `init_std=1.0, entropy=0.01` std 爆炸
   - 当前折中：`init_std=1.0, entropy=0.003`
10. viewer 增加 reward 曲线实时监控
11. 新增 `NAVRL_HANDOFF.md` 作为交接文档
12. 已将当前修改提交并推送到 fork 分支 `feature/go2-navrl-navigation`

## 13. 当前未解决问题和风险点

1. 成功率还没有稳定到接近 1.0
   - 当前最好实验峰值约 0.8+
   - 仍存在训练后期学坏或不稳定现象

2. `mean_std` 走势需要继续观察
   - 当前目标是不快速降到 0.1，也不爆炸到很大
   - 如果仍然 collapse，需要继续调 entropy、learning rate 或 KL

3. reward 权重仍需结合 viewer 检查
   - 重点看 `progress_towards_goal`
   - 重点看 `lidar_obstacle`
   - 重点看 `goal_bonus` 和 `termination_penalty` 的出现频率

4. collision 注释和实际代码不完全一致
   - 注释曾说 collision 不结束 episode
   - 当前 `lidar_collision` 实际会终止 episode

5. `vy=0` 是当前实验方案，不一定是最终方案
   - 如果后续发现绕障能力受限，可以对比恢复 3D action
   - 对比时不要同时改 reward、PPO 和 curriculum

6. USD 场景文件已经提交
   - 当前文件不大，约数百 KB 到 1.6 MB
   - 如果后续场景变大，应考虑 Git LFS 或外部资产管理

## 14. 建议下一步实验

继续坚持一次只改一个变量。

优先建议：

1. 跑完整当前配置
   - `init_std=1.0`
   - `entropy_coef=0.003`
   - `success_threshold=8`
   - `failure_threshold=4`
   - `vy=0`

2. 用 viewer 检查 reward 曲线
   - 失败 episode 中 `lidar_obstacle` 是否过强或过弱
   - `progress_towards_goal` 是否长期为负
   - `termination_penalty` 是否频繁出现

3. 如果 std 继续 collapse
   - 优先小幅增加 entropy，例如从 0.003 到 0.005
   - 或单独降低 learning rate
   - 不要同时改多个 PPO 参数

4. 如果 collision 仍多
   - 先确认 LiDAR 图和真实障碍距离一致
   - 再调整 `soft_threshold`、`hard_threshold` 或 `k_nearest`

5. 如果仍出现绕圈或原地转
   - 检查 action 输出、heading error、`vel_towards_goal`
   - 再决定是否恢复 `vy` 或增加 yaw/action regularization

## 15. 运行方式

普通 procedural terrain 训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Navigation-Go2-v0 \
  --num_envs 1024
```

加载 USD 场景训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Navigation-Go2-v0 \
  --num_envs 1024 \
  --load_scene source/robot_lab/data/environments/loco_navi_curriculum_flat_v1_baked_round_prims.usd
```

如果当前环境需要通过 IsaacLab launcher 启动，请在外层使用项目常用的 IsaacLab Python 启动方式。

启动 viewer：

```bash
python scripts/tools/lidar_viewer.py
```

启动 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/unitree_go2_navrl
```

## 16. Git 状态和交接说明

官方仓库 remote：

```text
upstream https://github.com/fan-ziqi/robot_lab.git
```

个人 fork remote：

```text
origin git@github.com:Bruce-Si/robot_lab.git
```

当前交接分支：

```text
feature/go2-navrl-navigation
```

当前主要提交：

```text
9c24a48 feat: add Go2 NavRL navigation task
```

如果修改了本文档或后续代码，提交并推送：

```bash
git add NAVRL_HANDOFF.md
git commit -m "docs: update navigation handoff"
git push
```

注意不要提交本地缓存：

```text
.claude/
.vscode/browse.vc.db*
scripts/source/
__pycache__/
logs/
outputs/
```

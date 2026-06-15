# Go2 NavRL Navigation 项目交接总结

更新时间：2026-06-15

## 1. 项目目标

本项目是在 RobotLab/IsaacLab 中训练 Unitree Go2 四足机器人完成二维导航避障任务。

高层策略负责根据目标状态和 LiDAR 输入输出导航速度指令；底层步态策略已经训练好并冻结，高层策略通过底层 locomotion policy 驱动机器人运动。整体设计参考 NavRL：

- 输入：低维目标/速度状态 + 2D LiDAR 深度图
- 输出：高层速度指令
- 训练算法：RSL-RL PPO
- 场景：50m x 50m cell，随机障碍物，课程学习逐步提高障碍难度

当前 task 注册名：

```bash
RobotLab-Navigation-Go2-v0
```

注册位置：

```text
source/robot_lab/robot_lab/tasks/manager_based/navigation/config/go2/__init__.py
```

## 2. 关键代码文件

导航任务主要新增/修改文件集中在：

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

debug viewer：

```text
scripts/tools/lidar_viewer.py
```

训练脚本做过一处改动：

```text
scripts/reinforcement_learning/rsl_rl/train.py
```

加入了 `--load_scene`，用于通过环境变量传入 USD 场景：

```text
NAVRL_USD_SCENE
NAVRL_OBSTACLE_JSON
```

注意：当前 `navigation/` 目录和 viewer 在 git 状态里是未跟踪文件，交接前需要确认是否加入版本控制。

## 3. 环境与场景

默认 procedural terrain 配置在：

```text
navigation_env_cfg.py
```

关键参数：

- terrain grid：8 x 8，共 64 个 unique layouts
- cell size：50m x 50m
- obstacle 数量从 15 增加到 90
- obstacle 高度：2.25m
- obstacle size 从 0.1m 到 2.0m
- `max_init_terrain_level=3`
- terrain curriculum 开启

如果使用 USD 场景，启动训练时通过 `--load_scene` 指定。配置里会：

- 关闭 procedural terrain
- 加载 USD 作为唯一场景几何
- 使用 `MultiMeshRayCasterCfg`
- 为 USD curriculum 创建 synthetic terrain-level state

边界相关：

- `terrain_out_of_bounds = None`
- 当前不惩罚越界
- 用户自己处理边界墙高度

## 4. 动作空间

当前使用 heading-locked 动作封装：

```text
mdp/heading_locked_action.py
```

高层策略 action 维度为 2：

```text
[vx_body, yaw_rate]
```

底层 locomotion policy 接收 3 维速度命令：

```text
[vx_body, 0.0, yaw_rate]
```

也就是说当前 `vy` 被锁为 0，机器人不能直接横移。想往左右走，需要先转向再前进。

做这个改动的原因：

- 之前允许 `vy` 时，机器人会出现类似“螃蟹步/绕圈接近目标”的行为
- reward 又鼓励朝向目标，可能导致高层策略在横移和朝向之间学出不自然动作
- 现在先用 `vy=0` 模式验证是否更容易稳定收敛

debug 箭头：

- 绿色箭头：实际发送给底层的线速度命令 `[vx, vy]`
- 蓝色箭头：机器人实际 body-frame 线速度
- 已修复之前把 `yaw_rate` 误画成横向速度的问题

## 5. 观测

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

其中：

- `goal_pos_body` 是目标在机器人 body frame 下的相对位置，3 维
- 归一化尺度固定为 50.0m
- `root_lin_vel_b` 是 body frame 线速度，3 维
- heading 使用 sin/cos 表达，2 维

之前曾使用“初始化时起点到终点距离”做归一化，后来改为固定 50.0m，避免不同 episode 的尺度随起终点变化。

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

- 远处/无障碍接近 0
- 近处障碍接近 1

注意：

- `/tmp/navrl_lidar.npy` 里保存的是 raw distance，单位 m，用于 viewer，不是 policy 输入的 proximity
- viewer 看到的是距离图，颜色条单位是 m

### 5.3 normalization

RSL-RL CNN model 配置中：

```python
obs_normalization = False
```

因此没有使用 RSL-RL 内置 observation normalization。

当前依赖手动归一化：

- goal 相对位置除以 50.0
- LiDAR 转为 0 到 1 的 proximity
- heading 使用 sin/cos
- 速度未额外归一化

## 6. 网络结构和 PPO

配置文件：

```text
config/go2/agents/rsl_rl_ppo_cfg.py
```

当前模型：

- `RslRlCNNModelCfg`
- actor/critic 都是 CNN + MLP
- actor/critic share CNN encoder
- activation：`elu`
- hidden dims：`[256, 256]`

CNN：

```python
output_channels=[4, 16, 32, 32]
kernel_size=[5, 5, 3, 3]
stride=[1, 2, 2, 2]
padding="zeros"
activation="elu"
flatten=True
```

PPO 当前关键参数：

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

关于 `mean_std` 的经验判断：

- 对当前 2D clipped action，健康范围大致希望最终在 0.3 到 0.6 左右
- 低于 0.15 通常说明探索过早收缩
- 高于 2 通常可疑，之前 entropy 太大时出现过 std 爆炸到 38+

## 7. Reward 设置

配置：

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

### 7.1 dense + sparse

当前是 dense + sparse：

- dense：朝目标速度、目标距离进展、LiDAR 避障、动作平滑
- sparse：到达目标 `goal_bonus=200`
- termination：失败终止 `-200`

到达目标 `goal_reached` 被设置为 `time_out=True`，也就是成功结束不吃 `termination_penalty`。

### 7.2 LiDAR obstacle penalty

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

这个函数返回负值或 0。由于配置里的 weight 是正数 2.0，所以最终加到总 reward 里仍然是惩罚。

之前发现过 soft penalty 写反的问题：如果 clamp 边界方向错，会导致越近障碍惩罚越弱或符号不符合预期。现在修正为：

```python
soft = torch.clamp(d - soft_threshold, min=hard_threshold - soft_threshold, max=0.0)
```

### 7.3 progress reward 注意事项

`progress_towards_goal` 会更新：

```python
env._nav_prev_goal_distance
```

所以 debug 代码不要重新调用 reward function，否则会改变训练状态。viewer 的 reward 曲线读取的是 `reward_manager._step_reward`，不重复调用 reward 函数。

## 8. Termination 设置

配置：

```text
navigation_env_cfg.py::TerminationsCfg
```

当前 termination：

- `time_out`
- `lidar_collision`
- `fallen_over`
- `goal_reached`

禁用：

- `terrain_out_of_bounds = None`

`lidar_collision`：

```python
mean(k nearest lidar distances) < body_radius
body_radius = 0.4
k_nearest = 5
```

`fallen_over`：

- grace period：50 steps
- 连续 40 steps 几乎不动
- body tilt > 75 deg

注意注释里有一句“collision only penalizes via reward, does NOT end episode”，但当前代码中 `lidar_collision` 实际是 termination，会结束 episode。这是交接时需要注意的注释/代码不一致点。

## 9. Curriculum

配置：

```text
navigation_env_cfg.py::CurriculumCfg
```

当前：

```python
success_threshold = 8
failure_threshold = 4
```

这属于“放慢课程学习”的版本。之前单独放慢 curriculum 后，训练明显改善，成功率峰值达到 0.8+，说明 curriculum 速度是主要影响因素之一。

curriculum 逻辑：

- 连续成功达到阈值后 level + 1
- 连续失败达到阈值后 level - 1
- 成功判定使用目标距离 `< 1.0m`

## 10. Debug viewer

文件：

```text
scripts/tools/lidar_viewer.py
```

环境侧会写出：

```text
/tmp/navrl_lidar.npy
/tmp/navrl_state.npy
/tmp/navrl_levels.npy
/tmp/navrl_rewards.npz
```

viewer 当前显示：

- 左侧：reward 各项曲线
- 左下：reward label 和最新数值
- 右上：状态信息、目标距离、heading error、速度、terrain level 分布
- 右下：LiDAR 距离图

reward 曲线：

- 只保留最近 100 个点
- 横坐标为当前选中 env 的 episode step
- env reset 后会清空上一段历史，避免把两个 episode 连起来
- reward 数据来自 `reward_manager._step_reward`

启动：

```bash
python scripts/tools/lidar_viewer.py
```

切换 env：

- 左右方向键：prev/next
- 输入数字后 Enter：跳转 env

注意：如果训练进程已经在跑，修改 `observations.py` 后不会被当前训练进程自动加载，需要重启训练/仿真进程。

## 11. 最近训练实验结论

日志目录：

```text
logs/rsl_rl/unitree_go2_navrl/
```

几个关键 run：

```text
2026-06-13_09-01-30_vy0_std1.0_entropy0.01
2026-06-14_17-35-16_vy0_std0.5_entropy0.001
2026-06-15_09-12-18_vy0_std1.0_entropy0.001
2026-06-15_14-26-08
```

已观察到的趋势：

1. `vy=0 + slow curriculum + init_std=1.0 + entropy=0.01`
   - 成功率峰值约 0.84
   - 但 `mean_std` 爆炸到 38+，探索过强，策略不稳定

2. `vy=0 + init_std=0.5 + entropy=0.001`
   - 效果变差
   - `mean_std` 下降到约 0.1
   - 成功率峰值约 0.58
   - 探索过早收缩

3. `vy=0 + init_std=1.0 + entropy=0.001`
   - 早期比低 std 版本好
   - 成功率曾到约 0.78
   - 但 std 仍持续下降，训练后期不够稳定

4. 当前折中方案
   - `init_std=1.0`
   - `entropy_coef=0.003`
   - 目标是避免 std 爆炸，同时也避免过早 collapse

当前最值得继续跑/观察的是第 4 类配置，也就是现在代码里的 PPO 配置。

## 12. 重要历史改动记录

按大致顺序：

1. 修正 LiDAR obstacle soft penalty 符号/ clamp 方向
2. 移除靠近 terrain 边界惩罚
3. 禁用 `terrain_out_of_bounds`
4. goal 相对位置归一化改成固定 50.0m
5. LiDAR policy 输入改成 proximity：`1 - distance / max_distance`
6. 高层 action 从 3D `[vx, vy, yaw_rate]` 改为 2D `[vx, yaw_rate]`，锁 `vy=0`
7. 修复 viewer/debug 速度箭头显示
8. 放慢 curriculum：`success_threshold=8, failure_threshold=4`
9. 尝试 PPO std/entropy：
   - `init_std=0.5, entropy=0.001` 过保守
   - `init_std=1.0, entropy=0.01` std 爆炸
   - 当前折中：`init_std=1.0, entropy=0.003`
10. viewer 增加 reward 曲线实时监控

## 13. 当前未解决问题 / 风险点

1. 成功率还没有稳定到接近 1.0
   - 当前最好的实验峰值约 0.8+
   - 仍存在训练后期学坏/不稳定现象

2. `mean_std` 走势需要继续看
   - 当前目标是维持在合理探索范围，不要快速降到 0.1，也不要爆炸

3. reward 权重仍需要用 viewer 检查
   - 特别是 `progress_towards_goal`、`lidar_obstacle`、`goal_bonus`、`termination_penalty` 的相对尺度
   - viewer 已能实时显示各项 weighted reward

4. 注释和实际 termination 有不一致
   - `navigation_env_cfg.py` 注释说 collision 不结束 episode
   - 实际 `lidar_collision` 是 termination

5. 当前高层锁 `vy=0` 只是实验方案
   - 如果后续发现路径绕障不够灵活，可以对比恢复 3D action
   - 但对比时不要同时改 reward/PPO/curriculum

6. 当前 navigation 目录是未跟踪文件
   - 交接/提交前必须确认 git add 范围
   - 不要误把 `.vscode/browse.vc.db*`、`.claude/` 等本地文件提交

## 14. 建议下一步实验

建议继续保持一次只改一个变量。

优先级：

1. 跑完整当前配置
   - `init_std=1.0`
   - `entropy_coef=0.003`
   - `success_threshold=8`
   - `failure_threshold=4`
   - `vy=0`

2. 用 viewer 检查 reward 曲线
   - 看失败 episode 中 `lidar_obstacle` 是否过强或过弱
   - 看 `progress_towards_goal` 是否长期为负
   - 看 `termination_penalty` 是否频繁出现

3. 如果 std 仍持续 collapse
   - 可以小幅增加 entropy，例如从 0.003 到 0.005
   - 或降低 learning rate / 调整 KL，但不要同时改

4. 如果 collision 仍多
   - 先检查 LiDAR 图和 `lidar_obstacle` 曲线是否匹配实际障碍距离
   - 再考虑调整 `soft_threshold`、`hard_threshold` 或 `k_nearest`

5. 如果路线仍绕圈或原地转
   - 检查 action 输出、heading error 和 `vel_towards_goal`
   - 再决定是否恢复 `vy` 或加入更明确的 yaw/action regularization

## 15. 运行参考

普通 procedural terrain 训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Navigation-Go2-v0 \
  --num_envs 1024
```

加载 USD 场景训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Navigation-Go2-v0 \
  --num_envs 1024 \
  --load_scene source/robot_lab/data/environments/loco_navi_v1.usd
```

如果当前项目通常通过 IsaacLab launcher 启动，则在外层加对应的 IsaacLab Python 启动方式。

viewer：

```bash
python scripts/tools/lidar_viewer.py
```

TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/unitree_go2_navrl
```


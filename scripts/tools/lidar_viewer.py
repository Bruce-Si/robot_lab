#!/usr/bin/env python3
"""LiDAR + state debug viewer. Arrows/type+Enter to switch env."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

LIDAR_PATH = "/tmp/navrl_lidar.npy"
STATE_PATH = "/tmp/navrl_state.npy"
LEVELS_PATH = "/tmp/navrl_levels.npy"
REWARDS_PATH = "/tmp/navrl_rewards.npz"
REWARD_HISTORY = 100

plt.ion()
fig = plt.figure(figsize=(16.5, 9.0))
fig.canvas.manager.set_window_title("Debug Viewer")

ax_reward = fig.add_axes([0.055, 0.59, 0.58, 0.34])
ax_reward.set_title("Weighted reward terms", fontsize=13)
ax_reward.set_xlabel("selected env episode step", fontsize=11)
ax_reward.set_ylabel("reward / s", fontsize=11)
ax_reward.tick_params(labelsize=10)
ax_reward.grid(True, alpha=0.25)
reward_lines = {}
total_reward_line = None

ax_episode_reward = fig.add_axes([0.055, 0.28, 0.58, 0.22])
ax_episode_reward.set_title("Episode cumulative reward", fontsize=12)
ax_episode_reward.set_xlabel("selected env episode step", fontsize=10)
ax_episode_reward.set_ylabel("return", fontsize=10)
ax_episode_reward.tick_params(labelsize=9)
ax_episode_reward.grid(True, alpha=0.25)
episode_total_line = None
episode_term_lines = {}

ax_reward_labels = fig.add_axes([0.055, 0.065, 0.58, 0.15])
ax_reward_labels.axis("off")

ax_info = fig.add_axes([0.69, 0.57, 0.28, 0.36])
ax_info.axis("off")
info_text = ax_info.text(0.02, 0.96, "", transform=ax_info.transAxes,
                         fontfamily="monospace", fontsize=12, va="top", linespacing=1.15)

ax_lidar = fig.add_axes([0.69, 0.13, 0.28, 0.33])
dummy = np.zeros((8, 72))
im = ax_lidar.imshow(dummy, aspect="auto", cmap="turbo_r",
                     vmin=0, vmax=4.0, origin="lower", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax_lidar, fraction=0.04, pad=0.02, shrink=0.6)
cbar.set_label("m", fontsize=11)
cbar.ax.tick_params(labelsize=10)
ax_lidar.tick_params(labelsize=10)
ax_lidar.set_xlabel("H beam (0=back 18=left 36=fwd 54=right)", fontsize=10)
ax_lidar.set_ylabel("V beam", fontsize=10)
t_lidar = ax_lidar.set_title("Waiting...", fontsize=12)

hint = fig.text(0.83, 0.055, "← →  prev/next     type#+Enter  jump     auto-refresh",
                ha="center", fontsize=9, color="gray")

cur_env = 0
last_lidar_mtime = 0
last_state_mtime = 0
last_levels_mtime = 0
last_rewards_mtime = 0
lidar_cache = None
state_cache = None
levels_cache = None
rewards_cache = None
typed = ""
n_envs = 0
reward_history = {
    "env": None,
    "step": [],
    "names": None,
    "values": [],
    "total": [],
    "episode_sums": [],
    "episode_total": [],
}


def _load():
    global last_lidar_mtime, last_state_mtime, last_levels_mtime, last_rewards_mtime
    global lidar_cache, state_cache, levels_cache, rewards_cache, n_envs
    if not os.path.exists(LIDAR_PATH):
        return None, None, None, None

    lm = os.path.getmtime(LIDAR_PATH)
    sm = os.path.getmtime(STATE_PATH) if os.path.exists(STATE_PATH) else 0
    lvm = os.path.getmtime(LEVELS_PATH) if os.path.exists(LEVELS_PATH) else 0
    rm = os.path.getmtime(REWARDS_PATH) if os.path.exists(REWARDS_PATH) else 0
    if (
        lm == last_lidar_mtime
        and sm == last_state_mtime
        and lvm == last_levels_mtime
        and rm == last_rewards_mtime
        and lidar_cache is not None
    ):
        return lidar_cache, state_cache, levels_cache, rewards_cache

    last_lidar_mtime = lm
    last_state_mtime = sm
    last_levels_mtime = lvm
    last_rewards_mtime = rm
    try:
        lidar_cache = np.load(LIDAR_PATH)
        state_cache = np.load(STATE_PATH) if sm > 0 else None
        levels_cache = np.load(LEVELS_PATH) if lvm > 0 else None
        if rm > 0:
            with np.load(REWARDS_PATH) as data:
                rewards_cache = {
                    "names": data["names"].astype(str).tolist(),
                    "values": data["values"].astype(np.float32),
                    "total": data["total"].astype(np.float32),
                    "episode_sums": data["episode_sums"].astype(np.float32)
                    if "episode_sums" in data
                    else None,
                    "episode_total": data["episode_total"].astype(np.float32)
                    if "episode_total" in data
                    else None,
                    "global_step": int(data["step"][0]),
                    "episode_steps": data["episode_steps"].astype(np.int64)
                    if "episode_steps" in data
                    else None,
                }
        else:
            rewards_cache = None
        n_envs = lidar_cache.shape[0]
    except Exception:
        return None, None, None, None
    return lidar_cache, state_cache, levels_cache, rewards_cache


def _update_reward_history(eid, rewards):
    if rewards is None or eid >= rewards["values"].shape[0]:
        return
    if rewards["episode_steps"] is not None and eid < rewards["episode_steps"].shape[0]:
        step = int(rewards["episode_steps"][eid])
    else:
        step = rewards["global_step"]
    if reward_history["env"] != eid:
        reward_history["env"] = eid
        reward_history["step"] = []
        reward_history["names"] = rewards["names"]
        reward_history["values"] = []
        reward_history["total"] = []
        reward_history["episode_sums"] = []
        reward_history["episode_total"] = []
    if reward_history["step"] and step < reward_history["step"][-1]:
        reward_history["step"] = []
        reward_history["values"] = []
        reward_history["total"] = []
        reward_history["episode_sums"] = []
        reward_history["episode_total"] = []
    if reward_history["step"] and reward_history["step"][-1] == step:
        return
    reward_history["names"] = rewards["names"]
    reward_history["step"].append(step)
    reward_history["values"].append(rewards["values"][eid].copy())
    reward_history["total"].append(float(rewards["total"][eid]))
    if rewards["episode_sums"] is not None and eid < rewards["episode_sums"].shape[0]:
        reward_history["episode_sums"].append(rewards["episode_sums"][eid].copy())
    else:
        reward_history["episode_sums"].append(None)
    if rewards["episode_total"] is not None and eid < rewards["episode_total"].shape[0]:
        reward_history["episode_total"].append(float(rewards["episode_total"][eid]))
    else:
        reward_history["episode_total"].append(None)
    if len(reward_history["step"]) > REWARD_HISTORY:
        reward_history["step"] = reward_history["step"][-REWARD_HISTORY:]
        reward_history["values"] = reward_history["values"][-REWARD_HISTORY:]
        reward_history["total"] = reward_history["total"][-REWARD_HISTORY:]
        reward_history["episode_sums"] = reward_history["episode_sums"][-REWARD_HISTORY:]
        reward_history["episode_total"] = reward_history["episode_total"][-REWARD_HISTORY:]


def _draw_rewards(eid, rewards):
    global total_reward_line
    _update_reward_history(eid, rewards)
    if not reward_history["step"]:
        ax_episode_reward.clear()
        ax_episode_reward.set_title("Episode cumulative reward", fontsize=12)
        ax_episode_reward.set_xlabel("selected env episode step", fontsize=10)
        ax_episode_reward.set_ylabel("return", fontsize=10)
        ax_episode_reward.tick_params(labelsize=9)
        ax_episode_reward.grid(True, alpha=0.25)
        ax_reward_labels.clear()
        ax_reward_labels.axis("off")
        ax_reward_labels.text(0.0, 0.65, "Waiting for rewards...",
                              transform=ax_reward_labels.transAxes, fontsize=11, va="center")
        return

    names = reward_history["names"]
    values = np.asarray(reward_history["values"], dtype=np.float32)
    total = np.asarray(reward_history["total"], dtype=np.float32)
    xs = np.asarray(reward_history["step"], dtype=np.int64)

    if total_reward_line is None:
        (total_reward_line,) = ax_reward.plot(xs, total, color="black", linewidth=1.8, label="total")
    else:
        total_reward_line.set_data(xs, total)

    for idx, name in enumerate(names):
        if name not in reward_lines:
            (line,) = ax_reward.plot(xs, values[:, idx], linewidth=1.0, label=name)
            reward_lines[name] = line
        else:
            reward_lines[name].set_data(xs, values[:, idx])

    for name, line in reward_lines.items():
        line.set_visible(name in names)

    ax_reward.relim()
    ax_reward.autoscale_view()
    if xs.size > 1:
        ax_reward.set_xlim(xs[0], xs[-1])

    _draw_episode_rewards(xs, names)

    latest = values[-1]
    latest_episode_total = reward_history["episode_total"][-1]
    total_label = "total"
    if latest_episode_total is not None:
        total_label = f"total | ep {latest_episode_total: .1f}"
    rows = [(total_label, float(total[-1]), total_reward_line.get_color())]
    rows.extend((name, float(latest[idx]), reward_lines[name].get_color()) for idx, name in enumerate(names))

    ax_reward_labels.clear()
    ax_reward_labels.axis("off")
    n_cols = 4 if len(rows) > 6 else 3
    n_per_col = int(np.ceil(len(rows) / n_cols))
    for ridx, (name, value, color) in enumerate(rows):
        col = ridx // n_per_col
        row = ridx % n_per_col
        x = col / n_cols + 0.015
        y = 0.82 - row * 0.42
        box = FancyBboxPatch(
            (x, y - 0.06), 0.018, 0.11,
            transform=ax_reward_labels.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=0.0,
            facecolor=color,
            clip_on=False,
        )
        ax_reward_labels.add_patch(box)
        ax_reward_labels.text(
            x + 0.026,
            y + 0.035,
            f"{name[:20]}",
            transform=ax_reward_labels.transAxes,
            fontfamily="monospace",
            fontsize=10.5,
            va="center",
            clip_on=False,
        )
        ax_reward_labels.text(
            x + 0.026,
            y - 0.075,
            f"{value: .3f}",
            transform=ax_reward_labels.transAxes,
            fontfamily="monospace",
            fontsize=10.5,
            va="center",
            clip_on=False,
        )


def _draw_episode_rewards(xs, names):
    global episode_total_line
    if not reward_history["episode_total"] or reward_history["episode_total"][-1] is None:
        ax_episode_reward.clear()
        ax_episode_reward.set_title("Episode cumulative reward", fontsize=12)
        ax_episode_reward.set_xlabel("selected env episode step", fontsize=10)
        ax_episode_reward.set_ylabel("return", fontsize=10)
        ax_episode_reward.tick_params(labelsize=9)
        ax_episode_reward.grid(True, alpha=0.25)
        ax_episode_reward.text(0.02, 0.75, "Restart training to export episode sums",
                               transform=ax_episode_reward.transAxes, fontsize=10)
        return

    episode_total = np.asarray(reward_history["episode_total"], dtype=np.float32)
    episode_sums = np.asarray(reward_history["episode_sums"], dtype=np.float32)

    if episode_total_line is None:
        (episode_total_line,) = ax_episode_reward.plot(
            xs, episode_total, color="black", linewidth=1.8, label="episode total"
        )
    else:
        episode_total_line.set_data(xs, episode_total)

    for idx, name in enumerate(names):
        if name not in episode_term_lines:
            (line,) = ax_episode_reward.plot(xs, episode_sums[:, idx], linewidth=0.9, alpha=0.75, label=name)
            episode_term_lines[name] = line
        else:
            episode_term_lines[name].set_data(xs, episode_sums[:, idx])

    for name, line in episode_term_lines.items():
        line.set_visible(name in names)

    ax_episode_reward.relim()
    ax_episode_reward.autoscale_view()
    if xs.size > 1:
        ax_episode_reward.set_xlim(xs[0], xs[-1])


def _draw(eid):
    lidar, state, levels, rewards = _load()
    if lidar is None:
        return 0
    eid = max(0, min(eid, n_envs - 1))

    # Flip LR (Isaac Sim Y=left, human Y=right) then roll fwd to center
    img = np.roll(np.fliplr(lidar[eid]), 36, axis=1)
    im.set_data(img)
    t_lidar.set_text(f"Env {eid}/{n_envs}   min dist = {img.min():.2f}m")

    # State
    if state is not None:
        s = state[eid]
        lvl = levels[eid] if levels is not None else -1
        pct = lambda c: f"{c / n_envs * 100:.0f}%"
        yaw_rate = s[5] if s.shape[0] > 5 else np.nan
        yaw_cmd = s[6] if s.shape[0] > 6 else np.nan
        fmt_rate = lambda v: f"{v:5.2f} rad/s" if np.isfinite(v) else "  n/a"
        state_lines = [
            f"ENV {eid}  (L{lvl})  step {s[4]:.0f}",
            f"",
            f"Target dist : {s[0]:5.1f} m",
            f"Heading err : {np.rad2deg(s[1]):4.0f} deg",
            f"Fwd vel     : {s[2]:5.2f} m/s",
            f"Lat vel     : {s[3]:5.2f} m/s",
            f"Yaw rate    : {fmt_rate(yaw_rate)}",
            f"Yaw cmd     : {fmt_rate(yaw_cmd)}",
            f"Min LiDAR   : {img.min():5.2f} m",
        ]
        if levels is not None:
            dist_lines = ["Level Dist", ""]
            for lv in range(8):
                cnt = (levels == lv).sum()
                bar = "|" * (int(pct(cnt).replace("%", "")) // 5)
                mark = " ◀" if lv == lvl else ""
                dist_lines.append(f"L{lv}: {bar} {pct(cnt)}{mark}")
            info_text.set_text("\n".join(state_lines + [""] + dist_lines))
        else:
            info_text.set_text("\n".join(state_lines))

    _draw_rewards(eid, rewards)

    fig.canvas.draw_idle()
    return eid


def on_key(event):
    global cur_env, typed
    lidar, _, _, _ = _load()
    if lidar is None:
        return

    if event.key == "left":
        cur_env = max(0, cur_env - 1)
    elif event.key == "right":
        cur_env = min(n_envs - 1, cur_env + 1)
    elif event.key == "enter":
        if typed.strip().isdigit():
            cur_env = max(0, min(int(typed.strip()), n_envs - 1))
        typed = ""
    elif event.key == "backspace":
        typed = typed[:-1]
    elif len(event.key) == 1 and event.key.isdigit():
        typed += event.key
    else:
        return

    _draw(cur_env)


fig.canvas.mpl_connect("key_press_event", on_key)
_draw(0)


def _auto_refresh():
    _draw(cur_env)


timer = fig.canvas.new_timer(interval=100)
timer.add_callback(_auto_refresh)
timer.start()

plt.show(block=True)

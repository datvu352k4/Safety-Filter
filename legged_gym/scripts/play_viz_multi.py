"""
play_viz_multi.py — Multi-robot visualization for research paper figures.
Now includes mixed terrain (flat/rough strips) and high robot spacing.
"""

import time
import os
import torch
import numpy as np
from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils import terrain_utils
from legged_gym.utils.terrain import Terrain

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_ROBOTS = 70  # Người dùng đã yêu cầu 500
COMMAND_RANDOM_EVERY = 250  # Reset lệnh sau mỗi 250 steps (~5 giây)


# ─── Custom Terrain (Strips) ────────────────────────────────────────────────
def _custom_curiculum(self):
    """Row i → terrain type: 0=flat, 1=rough (lặp lại mỗi 2 dòng)."""
    TYPE_SEQ = ["flat", "rough"]
    for j in range(self.cfg.num_cols):
        for i in range(self.cfg.num_rows):
            t = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            ttype = TYPE_SEQ[
                (i + j) % 2
            ]  # Đan chéo Phẳng/Rough theo kiểu bàn cờ (Checkerboard)
            if ttype == "flat":
                terrain_utils.pyramid_sloped_terrain(
                    t,
                    slope=0.0,
                    platform_size=self.platform_size,
                    terrain_type=self.type,
                )
            elif ttype == "rough":
                terrain_utils.random_uniform_terrain(
                    t,
                    min_height=-0.08,
                    max_height=0.08,  # +/- 8cm rough terrain
                    step=0.005,
                    downsampled_scale=0.2,
                    terrain_type=self.type,
                )
            self.add_terrain_to_map(t, i, j)


# Patch Terrain class
Terrain.curiculum = _custom_curiculum


def randomize_commands(env):
    num_envs = env.num_envs
    dev = env.device
    # vx: 0.4 m/s -> 1.2 m/s (tiến lên phía trước)
    env.commands[:, 0] = 0.4 + 0.8 * torch.rand(num_envs, device=dev)
    # vy: -0.4 m/s -> 0.4 m/s
    env.commands[:, 1] = -0.3 + 0.6 * torch.rand(num_envs, device=dev)
    # yaw: -0.8 rad/s -> 0.8 rad/s
    env.commands[:, 2] = -0.5 + 1.0 * torch.rand(num_envs, device=dev)
    print(f"[INFO] Commands randomized for {num_envs} robots.")


def override_configs(env_cfg, args):
    """Cấu hình để hiển thị nhiều robot trên bản đồ hỗn hợp rộng rãi."""
    env_cfg.env.num_envs = NUM_ROBOTS
    if "cts" in args.task:
        env_cfg.env.num_teacher = NUM_ROBOTS

    # Hiển thị tất cả envs
    env_cfg.viewer.rendered_envs_idx = list(range(NUM_ROBOTS))

    # Kích thước lưới địa hình để chứa 500 robot thưa thớt
    # VD: 125 hàng x 4 cột = 500 environments.
    # Mỗi ô địa hình rộng 12m x 12m để robot cách xa nhau.
    # Kích thước lưới địa hình thưa thớt, tự động scale theo số lượng robot
    # Tự động tính toán số hàng và cột để bản đồ gần với hình vuông nhất
    side = int(np.ceil(np.sqrt(NUM_ROBOTS)))
    num_cols = 4
    num_rows = 4

    env_cfg.terrain.num_rows = num_rows
    env_cfg.terrain.num_cols = num_cols
    env_cfg.terrain.terrain_length = 5.0
    env_cfg.terrain.terrain_width = 5.0
    env_cfg.terrain.border_size = 0.01
    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False

    # Tắt các randomization gây nhiễu cho visual
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.env.debug = (
        True  # Quan trọng: Để tắt chế độ tự động follow robot 0 trong class LeggedRobot
    )

    # Để camera tự do, không set cứng vị trí ban đầu nếu bạn muốn tự di chuyển
    # env_cfg.viewer.pos = [10.0, 10.0, 5.0]
    # env_cfg.viewer.lookat = [0.0, 0.0, 0.0]


def play(args):
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    # Khởi tạo môi trường
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load checkpoint
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Lấy history encoder (nếu có)
    encoder = None
    task = args.task
    if ppo_runner is not None and ("ts" in task or "cat" in task):
        for attr in [
            "alg.actor_critic.history_encoder",
            "alg.actor_critic.actor.history_encoder",
        ]:
            try:
                enc = ppo_runner
                for p in attr.split("."):
                    enc = getattr(enc, p)
                encoder = enc
                break
            except AttributeError:
                pass

    # Interaction Loop
    print(f"\n[VIZ] Starting Multi-Robot Swarm on Mixed Terrain...")
    print(f"[VIZ] Robots: {NUM_ROBOTS} | Spacing: {env_cfg.terrain.terrain_width}m")

    randomize_commands(env)
    frame_dt = 1 / 60.0

    for i in range(10 * int(env.max_episode_length)):
        t0 = time.perf_counter()

        if i % COMMAND_RANDOM_EVERY == 0 and i > 0:
            randomize_commands(env)

        obs_dict = env.get_observations()
        if isinstance(obs_dict, tuple):
            obs_buf, priv_obs, obs_history, critic_obs = obs_dict
            if encoder is not None:
                with torch.no_grad():
                    z_t = encoder(obs_history)
            actions = policy(obs_buf, obs_history)
            obs_results = env.step(actions.detach())
        else:
            obs = obs_dict
            actions = policy(obs.detach())
            obs_results = env.step(actions.detach())

        # Sync results
        if isinstance(obs_dict, tuple):
            obs_buf, priv_obs, obs_history, critic_obs, rews, dones, infos = obs_results
        else:
            obs, _, rews, dones, infos = obs_results

        elapsed = time.perf_counter() - t0
        if frame_dt - elapsed > 0:
            time.sleep(frame_dt - elapsed)


if __name__ == "__main__":
    args = get_args()
    if args.task is None:
        args.task = "go2_safety_terrain"
    play(args)

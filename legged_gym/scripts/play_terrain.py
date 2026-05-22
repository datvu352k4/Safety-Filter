"""
play_terrain.py — 2 terrain strips dọc theo trục X.

Layout (robot đi +X):
  Strip 0  X [0 - 10m]    FLAT          (mặt phẳng)
  Strip 1  X [10 - 20m]   ROUGH         (sỏi đá)

Friction thay đổi mỗi ZONE_LEN mét bên trong từng strip (xen kẽ hi/lo).
In: terrain type | friction | z_t 8D mỗi N bước.
"""

import time
import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils import terrain_utils
from legged_gym.utils.terrain import Terrain

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick

# ─── Layout hằng số ──────────────────────────────────────────────────────────
BORDER = 0.4  # Offset vật lý của lưới map Genesis
STRIP_LEN = 16.0  # mét mỗi strip (trục X) — bằng terrain_width
ZONE_LEN = 8.0  # mét mỗi friction sub-zone trong strip (x1-x2, x2-x3)
PRINT_EVERY = 10  # in mỗi N step

# Định nghĩa 2 strip: x0/x1 (world m), offset theo BORDER
STRIPS = [
    {
        "name": "flat",
        "label": "Mặt Phẳng",
        "x0": BORDER,
        "x1": BORDER + STRIP_LEN,
        "fhi": 1.0,
        "flo": 0.1,
    },
    {
        "name": "rough",
        "label": "Sỏi Đá",
        "x0": BORDER + STRIP_LEN,
        "x1": BORDER + 2 * STRIP_LEN,
        "fhi": 1.0,
        "flo": 0.1,
    },
]


# ─── Monkey-patch Terrain.curiculum ─────────────────────────────────────────
def _custom_curiculum(self):
    """Row i → terrain type: 0=flat, 1=rough (lặp lại)."""
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
            ttype = TYPE_SEQ[i % 2]
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
                    max_height=0.08,  # Level 8: +/- 8cm
                    step=0.005,
                    downsampled_scale=0.2,
                    terrain_type=self.type,
                )
            self.add_terrain_to_map(t, i, j)


# Patch trước khi make_env() gọi Terrain(cfg.terrain)
Terrain.curiculum = _custom_curiculum


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_strip(pos_x: float) -> dict:
    if pos_x < STRIPS[0]["x1"]:
        return STRIPS[0]
    return STRIPS[-1]


def target_friction(pos_x: float) -> float:
    """Friction xen kẽ hi/lo mỗi ZONE_LEN mét trong strip hiện tại."""
    s = get_strip(pos_x)
    within = (pos_x - s["x0"]) % STRIP_LEN
    even = int(within / ZONE_LEN) % 2 == 0
    return s["fhi"] if even else s["flo"]


def apply_friction(env, robot_idx: int, friction: float):
    """Cập nhật friction trong sim + obs buffer."""
    env.simulator._friction_values[robot_idx, 0] = friction
    dev = env.device
    ratios = torch.full((1, 1), friction, dtype=torch.float, device=dev)
    try:
        nlinks = env.simulator._robot.n_links
        env.simulator._robot.set_friction_ratio(
            ratios.repeat(1, nlinks),
            torch.arange(0, nlinks),
            [robot_idx],
        )
        if hasattr(env.simulator, "_gs_terrain"):
            env.simulator._gs_terrain.set_friction_ratio(ratios, envs_idx=[robot_idx])
    except Exception:
        pass


def print_debug(env, ridx, z_t, step):
    pos_x = env.simulator.base_pos[ridx, 0].item()
    vel_x = env.simulator.base_lin_vel[ridx, 0].item()
    cmd_x = env.commands[ridx, 0].item()
    h = env.simulator.base_pos[ridx, 2].item()
    try:
        fric = env.simulator._friction_values[ridx, 0].item()
    except Exception:
        fric = 0.0

    strip = get_strip(pos_x)
    print(
        f"[{step:05d}] X:{pos_x:6.2f}m | {strip['label']:<14} | "
        f"μ={fric:.3f} (GT) | h={h:.3f} cmd={cmd_x:+.2f} vel={vel_x:+.2f}"
    )

    if z_t is not None:
        zv = z_t[ridx].cpu().numpy()
        znorm = float(np.linalg.norm(zv))
        # Print all 8 dimensions of z_t
        z_str = " ".join([f"{v:+.2f}" for v in zv])
        print(f"         ‖z_t‖={znorm:.3f} | Latents: [{z_str}]")
    print()
    print()


# ─── Override configs ────────────────────────────────────────────────────────
def override_configs(env_cfg, args):
    """Cấu hình map 3 strip dọc theo X."""
    env_cfg.env.num_envs = 16
    if "cts" in args.task:
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = [0]

    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2  # 2 strip: flat/rough
        env_cfg.terrain.num_cols = 1  # 1 cột duy nhất
        env_cfg.terrain.terrain_length = STRIP_LEN  # = terrain_width → square cells
        env_cfg.terrain.terrain_width = STRIP_LEN  # 8.0m
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = True  # dùng _custom_curiculum
        env_cfg.terrain.selected = False
        env_cfg.terrain.max_init_terrain_level = 0  # bắt đầu ở row 0 (flat)
        env_cfg.env.debug_draw_terrain_height_points = False

    # Trọng tâm Row 0 (Flat) sinh ra ở X = BORDER + STRIP_LEN/2 = 5.0 + 16.0/2 = 13.0m
    # Nếu nó vẫn kẹt ở Rough do base_config, lùi mạnh thêm 16m (bằng hẳn 1 strip) về phía sau!
    env_cfg.init_state.pos = [-23.0, 0.0, 0.42]

    env_cfg.env.debug = True
    # Friction quản lý thủ công theo vị trí X — vô hiệu per-terrain range
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.05, 1.7]
    env_cfg.domain_rand.terrain_friction_ranges = {}  # tắt per-terrain friction

    env_cfg.commands.ranges.lin_vel_x = [-2.0, 2.0]
    env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.5, 1.5]
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False


# ─── Interaction loop ────────────────────────────────────────────────────────
def interaction_loop(env, policy, args, ppo_runner=None):
    logger = Logger(env.dt)
    ridx = 0
    task = args.task
    stop_state_log = 300
    stop_rew_log = env.max_episode_length + 1

    if "ts" in task or "cat" in task:
        obs_buf, priv_obs, obs_history, critic_obs = env.get_observations()
    else:
        obs = env.get_observations()

    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    # Lấy history encoder
    encoder = None
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
        if encoder:
            ndim = env.cfg.env.num_latent_dims
            print(f"\n[INFO] Encoder OK — z_t {ndim}D")

    # In header map
    print("=" * 68)
    print(f"  MAP 2 STRIPS (robot đi theo hướng +X, joystick lin_vel_x)")
    for s in STRIPS:
        print(
            f"  X [{s['x0']:5.1f} – {s['x1']:5.1f}m]  {s['label']:<14} "
            f"μ: {s['flo']:.1f} ↔ {s['fhi']:.1f} (mỗi {ZONE_LEN}m)"
        )
    print("=" * 68 + "\n")

    z_t = None
    prev_strip_name = None
    frame_dt = 1 / 60.0

    for i in range(10 * int(env.max_episode_length)):
        t0 = time.perf_counter()

        # Joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly * env.cfg.commands.ranges.lin_vel_x[1]
            env.commands[:, 1] = -joystick.lx * env.cfg.commands.ranges.lin_vel_y[1]
            env.commands[:, 2] = -joystick.rx * env.cfg.commands.ranges.ang_vel_yaw[1]

        if args.follow_robot:
            pos = env.simulator.base_pos[ridx].cpu().numpy() + np.array(
                env.cfg.viewer.pos, dtype=np.float32
            )
            lookat = env.simulator.base_pos[ridx].cpu().numpy() + np.array(
                env.cfg.viewer.lookat, dtype=np.float32
            )
            env.set_viewer_camera(pos, lookat)

        # ── Cập nhật friction theo vị trí X ────────────────────────────────
        pos_x = env.simulator.base_pos[ridx, 0].item()
        fric = target_friction(pos_x)
        apply_friction(env, ridx, fric)

        # ── Thông báo khi sang strip mới ────────────────────────────────────
        cur_strip = get_strip(pos_x)
        if cur_strip["name"] != prev_strip_name:
            print(f"\n{'─'*60}")
            print(
                f"  ⚡ SANG STRIP: {cur_strip['label'].upper()}"
                f"  (X={pos_x:.1f}m, μ={fric:.3f})"
            )
            print(f"{'─'*60}\n")
            prev_strip_name = cur_strip["name"]

        # ── Policy step ─────────────────────────────────────────────────────
        if "ts" in task or "cat" in task:
            if encoder is not None:
                with torch.no_grad():
                    z_t = encoder(obs_history)
            actions = policy(obs_buf, obs_history)
            obs_buf, priv_obs, obs_history, critic_obs, rews, dones, infos = env.step(
                actions.detach()
            )
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

        # ── Print ───────────────────────────────────────────────────────────
        if i % PRINT_EVERY == 0:
            print_debug(env, ridx, z_t, i)

        # ── Logger ──────────────────────────────────────────────────────────
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[ridx, 2].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.simulator.dof_pos[ridx, 2].item(),
                    "dof_vel": env.simulator.dof_vel[ridx, 2].item(),
                    "dof_torque": env.simulator.torques[ridx, 2].item(),
                    "command_x": env.commands[ridx, 0].item(),
                    "command_y": env.commands[ridx, 1].item(),
                    "command_yaw": env.commands[ridx, 2].item(),
                    "base_vel_x": env.simulator.base_lin_vel[ridx, 0].item(),
                    "base_vel_y": env.simulator.base_lin_vel[ridx, 1].item(),
                    "base_vel_z": env.simulator.base_lin_vel[ridx, 2].item(),
                    "base_vel_yaw": env.simulator.base_ang_vel[ridx, 2].item(),
                    "contact_forces_z": env.simulator.link_contact_forces[
                        ridx, env.simulator.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos.get("episode"):
                num_ep = torch.sum(env.reset_buf).item()
                if num_ep > 0:
                    logger.log_rewards(infos["episode"], num_ep)
        elif i == stop_rew_log:
            logger.print_rewards()

        elapsed = time.perf_counter() - t0
        if frame_dt - elapsed > 0:
            time.sleep(frame_dt - elapsed)


# ─── Export policy ───────────────────────────────────────────────────────────
def export_policy(alg_runner, path, args, env_cfg, train_cfg):
    task = args.task
    if "ts" in task or "cat" in task:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
    elif "ee" in task:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
    elif "dreamwaq" in task:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
    exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    print("Exported to:", path)


# ─── Main ────────────────────────────────────────────────────────────────────
def play(args):
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "Apr19_12-55-01_ts_terrain_genesis"
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    interaction_loop(env, policy, args, ppo_runner)


if __name__ == "__main__":
    args = get_args()
    play(args)

import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick


def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    task_name = args.task
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False

        # random uniform terrain
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.0,
            "max_height": 0.0,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }
        # slope
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_sloped_terrain",
        #     "slope": -0.4,
        #     "platform_size": 3.0,
        # }
        # stairs
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_stairs_terrain",
        #     "step_width": 0.31,
        #     "step_height": -0.1,
        #     "platform_size": 3.0,
        # }
        # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.discrete_obstacles_terrain",
        #     "max_height": 0.1,
        #     "min_size": 1.0,
        #     "max_size": 2.0,
        #     "num_rects": 20,
        #     "platform_size": 3.0,
        # }
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain",
        #                                   "amplitude": 0.1, "num_waves": 2}
        # stepping stones
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_size": 1.0, "max_height": 0.1,
        #                                   "stone_distance": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.gap_terrain",
        #     "gap_size": 0.2,
        #     "platform_size": 3.0,
        # }
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pit_terrain",
        #     "depth": 0.2,
        #     "platform_size": 3.0,
        # }

    env_cfg.env.debug = True
    env_cfg.domain_rand.friction_range = [0.05, 0.05]
    env_cfg.commands.ranges.lin_vel_x = [-2.0, 2.0]
    env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.5, 1.5]
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False


def print_debug_info(env, robot_index, z_t_val=None):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
        z_t_val (float, optional): Latent representation of friction. Defaults to None.
    """
    cmd_x = env.commands[robot_index, 0].item()
    base_height = env.simulator.base_pos[robot_index, 2].item()
    # Lấy vận tốc tiến/lùi thực tế của robot (Trục X)
    vel_x = env.simulator.base_lin_vel[robot_index, 0].item()

    # Lấy ma sát hiện tại (Genesis Simulator lưu trong _friction_values)
    try:
        friction = env.simulator._friction_values[robot_index, 0].item()
    except:
        friction = 0.0  # Phòng hờ lỗi nếu cấu trúc biến thay đổi

    # Format chuỗi z_t nếu có giá trị
    zt_str = f" | z_t: {z_t_val:.3f}" if z_t_val is not None else ""

    # In ra terminal
    print(
        f"Step: {env.episode_length_buf[robot_index].item():04d} | "
        f"Lệnh (Cmd): {cmd_x: .2f} m/s | "
        f"Thực tế (Act): {vel_x: .2f} m/s | "
        f"Độ cao (Z): {base_height:.3f} m | "
        f"Ma sát: {friction:.2f}{zt_str}"
    )


def interaction_loop(env, policy, args, ppo_runner=None):  # THÊM ppo_runner
    """Run interaction loop between environment and policy"""

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 2
    stop_state_log = 300
    stop_rew_log = env.max_episode_length + 1

    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif "ee" in task_name:
        estimator_features, _, _ = env.get_observations()
    elif "dreamwaq" in task_name:
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = (
            env.get_observations()
        )
    else:
        obs = env.get_observations()

    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    frame_dt = 1 / 60.0
    for i in range(10 * int(env.max_episode_length)):
        t_start = time.perf_counter()
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly * env.cfg.commands.ranges.lin_vel_x[1]
            env.commands[:, 1] = -joystick.lx * env.cfg.commands.ranges.lin_vel_y[1]
            env.commands[:, 2] = -joystick.rx * env.cfg.commands.ranges.ang_vel_yaw[1]

        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(
                env.cfg.viewer.pos, dtype=np.float32
            )
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(
                env.cfg.viewer.lookat, dtype=np.float32
            )
            env.set_viewer_camera(pos, lookat)

        z_t_val = None  # Biến chứa z_t để in log

        if "ts" in task_name or "cat" in task_name:
            # =========================================================
            # TÍNH TOÁN z_t BẰNG ENCODER TRƯỚC KHI STEP MÔI TRƯỜNG
            # =========================================================
            if ppo_runner is not None:
                with torch.no_grad():
                    try:
                        z_t = ppo_runner.alg.actor_critic.history_encoder(obs_history)
                    except AttributeError:
                        z_t = ppo_runner.alg.actor_critic.actor.history_encoder(
                            obs_history
                        )
                    z_t_val = z_t[robot_index, 0].item()

            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = (
                env.step(actions.detach())
            )
        elif "ee" in task_name:
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(
                actions.detach()
            )
        elif "waq" in task_name:
            actions = policy(obs_buf, obs_history)
            (
                obs_buf,
                privileged_obs_buf,
                obs_history,
                explicit_labels,
                next_states,
                rews,
                dones,
                infos,
            ) = env.step(actions.detach())
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

        # TRUYỀN z_t_val XUỐNG HÀM IN LOG
        print_debug_info(env, robot_index, z_t_val)

        # Update logger info (giữ nguyên phần code logger bên dưới)
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.simulator.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.simulator.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.simulator.torques[
                        robot_index, joint_index
                    ].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.simulator.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.simulator.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.simulator.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.simulator.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.simulator.link_contact_forces[
                        robot_index, env.simulator.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)

    print("Exported policy as jit script to: ", path)
    if args.export_onnx:
        print("Exported policy as onnx to: ", path)


def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level="warning",
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++ or python)
    # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
    #                         train_cfg.runner.load_run, 'exported')
    # export_policy(ppo_runner, path, args, env_cfg, train_cfg)

    interaction_loop(env, policy, args, ppo_runner)


if __name__ == "__main__":
    args = get_args()
    play(args)

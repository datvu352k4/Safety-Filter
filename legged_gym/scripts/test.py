import os
import time
import glob
import torch
import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.utils import get_args, task_registry, class_to_dict

# Import config để load mạng Safety
from legged_gym.envs.go2.go2_ts.go2_safety_config import Go2SafetyCfgPPO
import legged_gym.envs


def override_configs_test(env_cfg, args):
    """Cấu hình môi trường Test"""
    env_cfg.env.num_envs = 1
    env_cfg.viewer.rendered_envs_idx = [0]

    env_cfg.domain_rand.randomize_friction = True

    # =================================================================
    # >>> BẠN ĐỔI MA SÁT Ở ĐÂY ĐỂ TEST <<<
    # Băng trơn: [0.1, 0.1]
    # Sàn nhám : [1.5, 1.5]
    # =================================================================
    env_cfg.domain_rand.friction_range = [1.1, 1.1]

    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.terrain_kwargs = {
        "type": "terrain_utils.random_uniform_terrain",
        "min_height": 0.0,
        "max_height": 0.0,
        "step": 0.005,
        "downsampled_scale": 0.2,
    }
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.border_size = 1.0

    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 999999.0


def test_safety_loop(base_env, ll_policy, ll_model, safety_policy):
    obs_buf, _, obs_history, _ = base_env.reset()

    # Khởi tạo buffer (bắt đầu từ 0 để kiểm tra quá trình ramp-up)
    last_actions = torch.zeros(1, 3, device=base_env.device)
    alpha_scale = torch.zeros(1, 3, device=base_env.device)
    smoothed_cmd = torch.zeros(1, 3, device=base_env.device)

    current_friction = base_env.cfg.domain_rand.friction_range[0]

    print("\n" + "=" * 80)
    print(f">>> TEST BỘ LỌC AN TOÀN - FRICTION: {current_friction} <<<")
    print("=" * 80 + "\n")

    for i in range(1000):
        t_start = time.perf_counter()

        # 1. TRÍCH XUẤT THÔNG TIN CHO SAFETY FILTER (8 CHIỀU)
        try:
            z_t = ll_model.history_encoder(obs_history)
        except AttributeError:
            z_t = ll_model.actor.history_encoder(obs_history)

        base_lin_vel = base_env.simulator.base_lin_vel[:, :3]
        base_z = base_env.simulator.base_pos[:, 2].unsqueeze(1)

        # Ghép thành observation 8 chiều
        safety_obs = torch.cat([z_t, base_lin_vel, base_z, last_actions], dim=-1)

        # 2. SUY LUẬN TỪ MẠNG SAFETY
        with torch.no_grad():
            raw_actions = safety_policy(safety_obs)

        # 3. LÀM MƯỢT ALPHA (Asymmetric Low-pass Filter y hệt lúc Train)
        prev_alpha = alpha_scale.clone()
        raw_alpha = (torch.tanh(raw_actions) + 1.0) / 2.0

        delta = raw_alpha - prev_alpha
        rate = torch.where(
            delta > 0,
            torch.full_like(delta, 0.15),  # Tăng chậm (chống giật ga)
            torch.full_like(delta, 0.45),  # Giảm nhanh (phanh gấp khi nguy hiểm)
        )
        alpha_scale = torch.clamp(prev_alpha + rate * delta, 0.0, 1.0)
        last_actions = alpha_scale.clone()

        # 4. TẠO LỆNH USER VÀ ĐỔI HƯỚNG MỖI 150 BƯỚC (Giả lập Joystick/MPPI)
        # 0-149: Tiến thẳng; 150-299: Lách ngang; 300-449: Vừa đi vừa rẽ; 450-599: Phanh gấp
        mode = (i // 150) % 4
        if mode == 0:
            target_cmd = torch.tensor([[2.0, 0.0, 0.0]], device=base_env.device)
        elif mode == 1:
            target_cmd = torch.tensor([[0.0, 0.5, 0.0]], device=base_env.device)
        elif mode == 2:
            target_cmd = torch.tensor([[0.5, 0.0, 1.0]], device=base_env.device)
        else:
            target_cmd = torch.tensor([[0.0, 0.0, 0.0]], device=base_env.device)

        smoothed_cmd = 0.95 * smoothed_cmd + 0.05 * target_cmd

        # 5. TẠO LỆNH AN TOÀN VÀ GỬI XUỐNG LOCOMOTION
        safe_cmd = smoothed_cmd * alpha_scale
        base_env.commands[:, :3] = safe_cmd

        # 6. CHẠY VẬT LÝ LOCOMOTION
        with torch.no_grad():
            ll_actions = ll_policy(obs_buf, obs_history)
            obs_buf, _, obs_history, _, _, dones, _ = base_env.step(ll_actions)

            # IN KẾT QUẢ ĐỂ NGHIỆM THU
            zt_val = z_t[0, 0].item()
            alpha_x = alpha_scale[0, 0].item()
            alpha_y = alpha_scale[0, 1].item()
            alpha_w = alpha_scale[0, 2].item()

            sm_cmd_x = smoothed_cmd[0, 0].item()
            safe_x = safe_cmd[0, 0].item()
            vel_x = base_env.simulator.base_lin_vel[0, 0].item()
            bz = base_z[0, 0].item()

            if i % 10 == 0:
                print(
                    f"[Step {i:03d}] z_t: {zt_val:5.2f} | Alpha [X:{alpha_x:4.2f} Y:{alpha_y:4.2f} W:{alpha_w:4.2f}] | "
                    f"Cmd X: {sm_cmd_x:4.2f} -> Safe: {safe_x:4.2f} | VelX: {vel_x:5.2f} | Z: {bz:4.3f}m"
                )

            if dones[0]:
                print("\n>>> MÔI TRƯỜNG RESET <<<\n")
                last_actions.zero_()
                alpha_scale.zero_()
                smoothed_cmd.zero_()

        elapsed = time.perf_counter() - t_start
        if elapsed < 1 / 60.0:
            time.sleep(1 / 60.0 - elapsed)


def run_safety_test(args):
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    # --- 1. LOAD BASE ENV & LOCOMOTION ---
    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name="go2_ts")
    override_configs_test(base_env_cfg, args)

    base_env, _ = task_registry.make_env(name="go2_ts", args=args, env_cfg=base_env_cfg)

    base_train_cfg.runner.resume = True
    base_train_cfg.runner.load_run = "Mar29_23-07-53_ts_genesis"
    ll_runner, _ = task_registry.make_alg_runner(
        env=base_env, name="go2_ts", args=args, train_cfg=base_train_cfg
    )

    ll_policy = ll_runner.get_inference_policy(device=base_env.device)
    ll_model = ll_runner.alg.actor_critic
    ll_model.eval()

    # --- 2. LOAD SAFETY FILTER (Với Mock Env 8 chiều) ---
    safety_train_cfg = Go2SafetyCfgPPO()
    safety_train_cfg_dict = class_to_dict(safety_train_cfg)
    safety_train_cfg_dict["algorithm"].pop("encoder_lr", None)
    safety_train_cfg_dict["algorithm"].pop("num_encoder_epochs", None)

    class MockSafetyEnv:
        def __init__(self):
            self.num_obs = 8  # Đã sửa thành 8 chiều
            self.num_privileged_obs = (
                56  # Trùng khớp với Critic để không bị báo lỗi shape
            )
            self.num_actions = 3
            self.num_envs = 1
            self.device = base_env.device

        def reset(self):
            pass

    mock_env = MockSafetyEnv()
    from rsl_rl.runners import OnPolicyRunner

    safety_log_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", safety_train_cfg.runner.experiment_name
    )
    safety_runner = OnPolicyRunner(
        env=mock_env,
        train_cfg=safety_train_cfg_dict,
        log_dir=safety_log_dir,
        device=base_env.device,
    )

    checkpoints = glob.glob(os.path.join(safety_log_dir, "model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Không tìm thấy file model trong {safety_log_dir}")

    # Sắp xếp để lấy file checkpoint mới nhất
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    resume_path = checkpoints[-1]
    print(f"\n✅ Đã load Safety Model từ: {resume_path}")
    safety_runner.load(resume_path)
    safety_policy = safety_runner.get_inference_policy(device=base_env.device)

    # --- 3. CHẠY VÒNG LẶP TEST ---
    test_safety_loop(base_env, ll_policy, ll_model, safety_policy)


if __name__ == "__main__":
    args = get_args()
    run_safety_test(args)

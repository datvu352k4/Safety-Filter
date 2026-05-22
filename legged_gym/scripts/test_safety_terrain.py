import os
import time
import glob
import torch
import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.utils import get_args, task_registry, class_to_dict

# Import config để load mạng Safety
from legged_gym.envs.go2.go2_ts_terrain.go2_safety_terrain_config import (
    Go2SafetyTerrainCfgPPO,
)
import legged_gym.envs

# =====================================================================
# >>> CẤU HÌNH BÀI TEST Ở ĐÂY <<<
# =====================================================================
TEST_TERRAIN_TYPE = "flat"  # Chọn 1 trong 2: "flat", "rough"
TEST_FRICTION = 1.1  # Băng trơn: 0.1,  Sàn nhám test an toàn: 0.4, Đá bám: 1.0
BYPASS_SAFETY = (
    False  # Cài thành True để TẮT Safety Filter (ép alpha = 1.0) và xem robot trượt té
)
# =====================================================================


def override_configs_test(env_cfg, args):
    """Cấu hình môi trường Test"""
    env_cfg.env.num_envs = 1
    env_cfg.viewer.rendered_envs_idx = [0]

    env_cfg.domain_rand.randomize_friction = True

    # Ma sát tuỳ chỉnh lúc Test
    env_cfg.domain_rand.friction_range = [TEST_FRICTION, TEST_FRICTION]
    if hasattr(env_cfg.domain_rand, "terrain_friction_ranges"):
        env_cfg.domain_rand.terrain_friction_ranges = {
            1.0: [TEST_FRICTION, TEST_FRICTION],
        }

    if TEST_TERRAIN_TYPE == "flat":
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.terrain_kwargs = None
    else:  # Mặc định là "rough"
        env_cfg.terrain.mesh_type = "heightfield"
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.07,
            "max_height": 0.07,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }

    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True  # Ép sinh map theo đúng tuỳ chọn phía trên
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.border_size = 5.0  # Tăng border map để có không gian chạy tay cầm

    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 999999.0


def test_safety_loop(base_env, ll_policy, ll_model, safety_policy, args=None):
    """
    Vòng lặp Test có thể tương tác (hỗ trợ Joystick)
    """
    obs_buf, _, obs_history, _ = base_env.reset()

    last_actions = torch.ones(
        (base_env.num_envs, 3), device=base_env.device, dtype=torch.float
    )
    smoothed_cmd = torch.zeros(
        (base_env.num_envs, 3), device=base_env.device, dtype=torch.float
    )

    # Cấu hình Joystick
    joystick = None
    if getattr(args, "use_joystick", False):
        try:
            from legged_gym.scripts.joystick import Joystick

            joystick = Joystick(joystick_type=getattr(args, "joystick_type", "xbox"))
            print("[Info] Đã kết nối Joystick để điều khiển bằng tay.")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi Joystick: {e}. Quay về chế độ chạy tự động.")
            joystick = None

    print(
        f"\n================================================================================"
    )
    print(
        f">>> TEST SAFETY FILTER | TERRAIN: {TEST_TERRAIN_TYPE.upper()} | FRICTION: {TEST_FRICTION} <<<"
    )
    print("=" * 80 + "\n")

    for i in range(5000):  # Tăng số step lên để đủ thời gian lái
        t_start = time.perf_counter()

        # 1. TRÍCH XUẤT THÔNG TIN CHO SAFETY FILTER (15 CHIỀU)
        try:
            z_t = ll_model.history_encoder(obs_history)
        except AttributeError:
            z_t = ll_model.actor.history_encoder(obs_history)

        base_z = base_env.simulator.base_pos[:, 2].unsqueeze(1)

        # Ghép thành observation 6 chiều (z_t(3) + last_actions(3))
        safety_obs = torch.cat([z_t, last_actions], dim=-1)

        # 2. SUY LUẬN TỪ MẠNG SAFETY
        with torch.no_grad():
            raw_actions = safety_policy(safety_obs)
            if BYPASS_SAFETY:
                raw_actions = (
                    torch.ones_like(raw_actions) * 100.0
                )  # Tạo alpha raw thật lớn để sigmoid -> 1.0

        # 3. ÁP DỤNG ALPHA TRỰC TIẾP (End-to-End learned smoothing)
        # Bỏ bộ lọc LPF cứng để đồng bộ 100% với môi trường huấn luyện Baseline mới
        alpha_scale = (torch.tanh(raw_actions) + 1.0) / 2.0
        alpha_scale = alpha_scale.clamp(0.2, 1.0)  # [0.2, 1.0] per report baseline
        last_actions = alpha_scale.clone()

        # 4. TẠO LỆNH USER (TỰ ĐỘNG HOẶC CAMERA/JOYSTICK)
        if joystick:
            joystick.update()
            # Mở khóa toàn bộ công suất thay vì bị giới hạn bởi config ban đầu (0.5m/s)
            vx_max = 2.0
            vy_max = 1.0
            yaw_max = 1.5
            target_cmd = torch.tensor(
                [
                    [
                        -joystick.ly * vx_max,
                        -joystick.lx * vy_max,
                        -joystick.rx * yaw_max,
                    ]
                ],
                device=base_env.device,
            )
        else:
            # B0 (0-200): Chạy thẳng max tốc (Test Alpha X)
            # B1 (201-400): Xoay Yaw tại chỗ (Test Alpha W - CỨU YAW)
            # B2 (401-600): Đi lùi + Xoay (Test Mixed)
            # B3 (601-800): Chạy ngang (Test Alpha Y)
            # B4 (801-1000): Phanh gấp
            step_cycle = 200
            mode = (i // step_cycle) % 5
            if mode == 0:
                target_cmd = torch.tensor([[2.0, 0.0, 0.0]], device=base_env.device)
            elif mode == 1:
                target_cmd = torch.tensor([[0.0, 0.0, 1.5]], device=base_env.device)
            elif mode == 2:
                target_cmd = torch.tensor([[-1.0, 0.0, 1.0]], device=base_env.device)
            elif mode == 3:
                target_cmd = torch.tensor([[0.0, 1.5, 0.0]], device=base_env.device)
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
            zt_vals = z_t[0].cpu().tolist()  # 3 chiều đầy đủ
            zt_str = " ".join(f"{v:+.2f}" for v in zt_vals)
            alpha_x = alpha_scale[0, 0].item()
            alpha_y = alpha_scale[0, 1].item()
            alpha_w = alpha_scale[0, 2].item()

            sm_cmd_x = smoothed_cmd[0, 0].item()
            sm_cmd_w = smoothed_cmd[0, 2].item()
            safe_x = safe_cmd[0, 0].item()
            safe_w = safe_cmd[0, 2].item()
            vel_x = base_env.simulator.base_lin_vel[0, 0].item()
            vel_w = base_env.simulator.base_ang_vel[0, 2].item()
            bz = base_z[0, 0].item()

            # ĐO LƯỜNG ĐỘ TRƯỢT CHÂN THỰC TẾ ĐỂ CHUẨN HÓA DEADZONE
            feet_vel_xy = base_env.simulator._feet_vel[:, :, :2]
            feet_contact_force = base_env.simulator._link_contact_forces[
                :, base_env.simulator._feet_indices, :
            ]
            in_contact = (feet_contact_force.norm(dim=-1) > 5.0).float()
            slip = (feet_vel_xy * in_contact.unsqueeze(-1)).norm(dim=-1)
            total_slip = slip.sum(dim=-1)[0].item()

            if i % 10 == 0:
                print(f"[Step {i:03d}] z_t: [{zt_str}]")
                print(
                    f"         Alpha [X:{alpha_x:4.2f} Y:{alpha_y:4.2f} W:{alpha_w:4.2f}] | "
                    f"Cmd X: {sm_cmd_x:4.2f} -> Safe: {safe_x:4.2f} | VelX: {vel_x:5.2f} | "
                    f"Yaw: {vel_w:+.2f} | Z: {bz:4.3f}m | Trượt/Slip: {total_slip:5.3f} m/s"
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
    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name="go2_ts_terrain")
    override_configs_test(base_env_cfg, args)

    base_env, _ = task_registry.make_env(
        name="go2_ts_terrain", args=args, env_cfg=base_env_cfg
    )

    base_train_cfg.runner.resume = True
    base_train_cfg.runner.load_run = "Apr19_12-55-01_ts_terrain_genesis"
    ll_runner, _ = task_registry.make_alg_runner(
        env=base_env, name="go2_ts_terrain", args=args, train_cfg=base_train_cfg
    )

    ll_policy = ll_runner.get_inference_policy(device=base_env.device)
    ll_model = ll_runner.alg.actor_critic
    ll_model.eval()

    # --- 2. LOAD SAFETY FILTER (Với Mock Env) ---
    safety_train_cfg = Go2SafetyTerrainCfgPPO()
    safety_train_cfg_dict = class_to_dict(safety_train_cfg)
    safety_train_cfg_dict["algorithm"].pop("encoder_lr", None)
    safety_train_cfg_dict["algorithm"].pop("num_encoder_epochs", None)

    class MockSafetyEnv:
        def __init__(self):
            self.num_obs = 6  # Actor: z_t(3) + last_alpha(3)
            self.num_privileged_obs = 72  # Critic: 72D per report baseline
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

    # Sắp xếp để lấy file checkpoint mới nhất được train (tránh dính file cũ từ các đợt train cũ)
    checkpoints.sort(key=os.path.getmtime)

    resume_path = checkpoints[-1]
    print(f"\n✅ Đã load Safety Model từ: {resume_path}")
    safety_runner.load(resume_path)
    safety_policy = safety_runner.get_inference_policy(device=base_env.device)

    # --- 3. CHẠY VÒNG LẶP TEST ---
    test_safety_loop(base_env, ll_policy, ll_model, safety_policy, args)


if __name__ == "__main__":
    args = get_args()
    run_safety_test(args)

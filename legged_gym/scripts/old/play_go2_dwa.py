import os
import sys
import types
import time
import importlib.util
import math
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

from legged_gym import *
from legged_gym.envs import *
import heapq
import torch.nn.functional as F
import yaml
import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.math_utils import quat_rotate_inverse, quat_apply, get_euler_xyz
from legged_gym.simulator.genesis_simulator import GenesisSimulator


def angle_range_corrector(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Import DWA from local folder
sys.path.append(os.path.join(LEGGED_GYM_ROOT_DIR, "dwa"))
from dwa import DWA, Obstacle

# ==============================================================================
# ─── 1. CẤU HÌNH HỆ THỐNG ──────────────────────────────────────────────────────
# ==============================================================================


class Go2MeshConfig:
    # ── Thông tin Robot & Policy ──
    LOG_DIR = "/home/datvu/LeggedGym-Ex/logs/go2_rough_terrain/Apr10_16-52-15_ts_terrain_genesis"
    CHECKPOINT = 1500

    # ── Vật thể 3D (.obj) ──
    OBJ_FILES = [
        {
            "path": "test_map/map1/map1.obj",
            "pos": (0.0, 0.0, 0.0),
            "euler": (90, 0, -90),
            "scale": 1.0,
            "fixed": True,
        },
    ]

    # ── Cấu hình Robot & Mô phỏng ──
    INIT_POS = [0.0, 0.0, 0.42]
    DEFAULT_FRICTION = 1.0

    # Định nghĩa nhiều vùng ma sát: [x_min, x_max, y_min, y_max, friction]
    FRICTION_ZONES = [
        [2.0, 7.5, 5.0, 10.0, 0.1],
        [9.0, 13.0, 6.0, 13.0, 0.1],  # Map1
    ]

    # FRICTION_ZONES = [
    #     [2.0, 7.5, 5.0, 10.0, 0.1],
    #     [10.0, 13.0, 6.0, 13.0, 0.1],  # Map2
    # ]

    # ── Điều hướng (DWA Navigation) ──
    USE_DWA = True
    GOAL_POS = [13.0, 15.0]  # Map1
    # GOAL_POS = [13.0, 15.0] # Map2

    MAP_YAML_PATH = "slam_map/map1/map1.yaml"

    # Tham số DWA
    DWA_WEIGHT_ANGLE = 15.0  # Khớp hướng tương đối
    DWA_WEIGHT_VEL = 5.0  # Ưu tiên tốc độ trên đường thẳng
    DWA_WEIGHT_OBS = 5.0  # Ưu tiên né vật cản
    DWA_WEIGHT_PATH = 20.0  # ƯU TIÊN SỐ 1: BÁM SÁT ĐƯỜNG A*
    DWA_WEIGHT_LAT = 0.5  # Phạt đi ngang nhẹ
    DWA_MARGIN = 0.15  # Lề an toàn
    DWA_DELTA_VEL = 0.05
    DWA_DELTA_ANG_VEL = 0.05
    DWA_PRE_STEP = 30  # 0.9s (30 * 0.03)

    # Giới hạn vận tốc DWA (Khi Alpha = 1.0)
    V_MAX_MAX = [2.0, 1.0, 1.5]
    V_MIN_MAX_BACK = -0.2

    # ── Safety Filter ──
    USE_SAFETY_FILTER = True
    SAFETY_MODEL_PATH = "logs/go2_safety_terrain/model_800.pt"

    # ── Hiển thị ──
    SHOW_VIEWER = False


# ==============================================================================
# ─── 2. HÀM BỔ TRỢ & CẤU TRÚC ĐIỀU HƯỚNG ─────────────────────────────────────
# ==============================================================================


# 2.2 Thuật toán A* Planner & Smoother
class AStarPlanner:
    def __init__(self, map_yaml_path, inflation_radius=0.7):
        if not os.path.exists(map_yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file bản đồ: {map_yaml_path}")
        with open(map_yaml_path, "r") as f:
            map_info = yaml.safe_load(f)
        self.resolution = map_info["resolution"]
        self.origin = map_info["origin"]
        pgm_path = os.path.join(os.path.dirname(map_yaml_path), map_info["image"])
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        occ = (255.0 - img) / 255.0
        self.grid = (occ > map_info["occupied_thresh"]).astype(np.uint8)
        inf_pixels = int(math.ceil(inflation_radius / self.resolution))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (inf_pixels * 2 + 1, inf_pixels * 2 + 1)
        )
        self.grid = cv2.dilate(self.grid, kernel)
        self.height, self.width = self.grid.shape

    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = self.height - 1 - int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = self.origin[0] + gx * self.resolution
        y = self.origin[1] + (self.height - 1 - gy) * self.resolution
        return x, y

    def plan(self, start_w, goal_w):
        start = self.world_to_grid(*start_w)
        goal = self.world_to_grid(*goal_w)
        if (
            not (0 <= start[0] < self.width and 0 <= start[1] < self.height)
            or self.grid[start[1], start[0]]
        ):
            return None
        if (
            not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height)
            or self.grid[goal[1], goal[0]]
        ):
            return None
        neighbors = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(self.grid_to_world(*start))
                path.reverse()
                return path
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if 0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height:
                    if self.grid[nxt[1], nxt[0]] == 0:
                        cost = math.hypot(dx, dy)
                        tentative_g = g_score[current] + cost
                        if nxt not in g_score or tentative_g < g_score[nxt]:
                            came_from[nxt] = current
                            g_score[nxt] = tentative_g
                            heapq.heappush(
                                open_set, (tentative_g + heuristic(nxt, goal), nxt)
                            )
        return None


def simple_smoother(path, weight_data=0.2, weight_smooth=0.4, tolerance=0.01):
    if not path or len(path) <= 2:
        return path
    new_path = [[p[0], p[1]] for p in path]
    change = tolerance
    iters = 0
    while change >= tolerance and iters < 100:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(2):
                aux = new_path[i][j]
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (
                    new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j]
                )
                change += abs(aux - new_path[i][j])
        iters += 1
    return new_path


# 2.4 Mạng Safety Filter (Dự đoán độ an toàn)
class SafetyFilter(nn.Module):
    def __init__(self, in_dim=6, hidden_dims=[256, 128], out_dim=3):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ELU())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor(x)


def load_safety_filter(checkpoint_path, device):
    print(f"[Safety] Đang nạp mạng Safety từ: {checkpoint_path}")
    model = SafetyFilter(in_dim=6, hidden_dims=[256, 128], out_dim=3).to(device)
    data = torch.load(checkpoint_path, map_location=device)
    state_dict = data["model_state_dict"] if "model_state_dict" in data else data
    actor_state_dict = {
        k[6:]: v for k, v in state_dict.items() if k.startswith("actor.")
    }
    model.actor.load_state_dict(actor_state_dict)
    model.eval()
    return model


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def apply_stability_patches(env, args):
    env.simulator._env_origins[:] = 0
    if hasattr(env.simulator, "_terrain_origins"):
        env.simulator._terrain_origins[:] = 0

    def custom_reset_root(self_env, env_ids):
        sim = self_env.simulator
        base_pos = (
            sim.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            + sim._env_origins[env_ids]
        )
        base_quat = sim.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        z_vel = torch.zeros((len(env_ids), 3), device=self_env.device)
        sim.reset_root_states(env_ids, base_pos, base_quat, z_vel, z_vel)

    env._reset_root_states = types.MethodType(custom_reset_root, env)

    def custom_terrain_info(self_sim):
        if self_sim._cfg.terrain.mesh_type == "plane":
            if hasattr(self_sim, "_height_around_feet"):
                self_sim._height_around_feet[:] = 0.0
            if hasattr(self_sim, "_normal_vector_around_feet"):
                self_sim._normal_vector_around_feet[:, 0::3] = 0.0
                self_sim._normal_vector_around_feet[:, 1::3] = 0.0
                self_sim._normal_vector_around_feet[:, 2::3] = -1.0

    env.simulator._calc_terrain_info_around_feet = types.MethodType(
        custom_terrain_info, env.simulator
    )

    def dummy_set_cam(self_env, pos=None, lookat=None):
        pass

    env.set_viewer_camera = types.MethodType(dummy_set_cam, env)


# ==============================================================================
# ─── 3. THỰC THI CHƯƠNG TRÌNH ──────────────────────────────────────────────────
# ==============================================================================


def play(args):
    cfg = Go2MeshConfig()
    args.headless = not cfg.SHOW_VIEWER
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )

    config_path = os.path.join(cfg.LOG_DIR, "go2_ts_terrain_config.py")
    env_path = os.path.join(cfg.LOG_DIR, "go2_ts_terrain.py")
    config_mod = import_module_from_path("log_config", config_path)
    env_mod = import_module_from_path("log_env", env_path)

    task_registry.register(
        "go2_mesh_dwa",
        env_mod.Go2TSTerrain,
        config_mod.Go2TSTerrainCfg(),
        config_mod.Go2TSTerrainCfgPPO(),
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name="go2_mesh_dwa")

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 3600.0
    env_cfg.seed = -1  # Kích hoạt Random Seed mặc định
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.rendered_envs_idx = [0]

    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.init_state.pos = cfg.INIT_POS
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.heading_command = False  # QUAN TRỌNG: Tắt bám hướng mặc định

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = os.path.basename(cfg.LOG_DIR)
    train_cfg.runner.checkpoint = cfg.CHECKPOINT

    orig_create_sim = GenesisSimulator._create_sim

    def patched_create_sim(self_sim):
        orig_create_sim(self_sim)
        scene_inst = self_sim._scene
        orig_build = scene_inst.build

        def patched_build(*args, **kwargs):
            robot_ent = getattr(self_sim, "_robot", None)
            if robot_ent is not None:
                pattern = gs.sensors.SphericalPattern(
                    fov=(360.0, 0.0), n_points=(144, 1)
                )
                self_sim.lidar_sensor = scene_inst.add_sensor(
                    gs.sensors.Lidar(
                        pattern=pattern,
                        entity_idx=robot_ent.idx,
                        pos_offset=(0.0, 0.0, 0.15),
                        max_range=100.0,
                        min_range=0.3,
                        draw_debug=cfg.SHOW_VIEWER,
                    )
                )
            return orig_build(*args, **kwargs)

        scene_inst.build = patched_build
        for obj in cfg.OBJ_FILES:
            p = (
                os.path.join(LEGGED_GYM_ROOT_DIR, obj["path"])
                if not os.path.isabs(obj["path"])
                else obj["path"]
            )
            if os.path.exists(p):
                entity = scene_inst.add_entity(
                    gs.morphs.Mesh(
                        file=p,
                        pos=obj["pos"],
                        euler=obj["euler"],
                        scale=obj["scale"],
                        fixed=obj["fixed"],
                    )
                )
                entity.set_friction(cfg.DEFAULT_FRICTION)

    GenesisSimulator._create_sim = patched_create_sim

    env, _ = task_registry.make_env(name="go2_mesh_dwa", args=args, env_cfg=env_cfg)
    apply_stability_patches(env, args)
    if hasattr(env.simulator, "_gs_terrain"):
        env.simulator._gs_terrain.set_friction(cfg.DEFAULT_FRICTION)
    env.simulator._friction_values[:] = cfg.DEFAULT_FRICTION

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name="go2_mesh_dwa", args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    safety_filter = load_safety_filter(
        os.path.join(LEGGED_GYM_ROOT_DIR, cfg.SAFETY_MODEL_PATH), env.device
    )
    last_safety_alpha = torch.ones(1, 3, device=env.device)
    V_MAX_MAX = torch.tensor(cfg.V_MAX_MAX, device=env.device)

    # DWA Initialization
    dwa_solver = DWA(samplingtime=0.03) if cfg.USE_DWA else None
    if dwa_solver is not None:
        dwa_solver.margin = cfg.DWA_MARGIN
        dwa_solver.weight_angle = cfg.DWA_WEIGHT_ANGLE
        dwa_solver.weight_vel = cfg.DWA_WEIGHT_VEL
        dwa_solver.weight_obs = cfg.DWA_WEIGHT_OBS
        dwa_solver.weight_path = cfg.DWA_WEIGHT_PATH
        dwa_solver.weight_lat = cfg.DWA_WEIGHT_LAT
        dwa_solver.delta_vel = 0.05
        dwa_solver.delta_lat_vel = 0.1
        dwa_solver.delta_ang_vel = 0.05
        dwa_solver.pre_step = cfg.DWA_PRE_STEP
        dwa_solver.simu_robot.lim_max_vel = V_MAX_MAX[0].item()
        dwa_solver.simu_robot.lim_min_vel = cfg.V_MIN_MAX_BACK
        dwa_solver.simu_robot.lim_max_lat_vel = V_MAX_MAX[1].item()
        dwa_solver.simu_robot.lim_min_lat_vel = -V_MAX_MAX[1].item()
        dwa_solver.simu_robot.lim_max_accel = 5.0
        dwa_solver.simu_robot.lim_max_ang_accel = 5.0

    map_yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.MAP_YAML_PATH)
    astar = AStarPlanner(map_yaml_path)
    global_path = astar.plan(cfg.INIT_POS[:2], cfg.GOAL_POS)
    if global_path:
        global_path = simple_smoother(global_path)
    goal_pos_numpy = np.array([cfg.GOAL_POS[0], cfg.GOAL_POS[1], 0.25]).reshape(1, 3)
    path_numpy = (
        np.array([[p[0], p[1], 0.1] for p in global_path]) if global_path else None
    )
    encoder = getattr(
        ppo_runner.alg.actor_critic,
        "history_encoder",
        getattr(ppo_runner.alg.actor_critic.actor, "history_encoder", None),
    )

    joystick = Joystick(joystick_type=args.joystick_type) if args.use_joystick else None

    def get_friction_at(x, y, cfg):
        for zone in cfg.FRICTION_ZONES:
            if zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]:
                return zone[4]
        return cfg.DEFAULT_FRICTION

    # VÔ HIỆU HÓA TẤT CẢ CÁC TRÌNH GHI ĐÈ LỆNH ĐỂ LÀM CHỦ ROBOT
    env._resample_commands = lambda *args, **kwargs: None
    if hasattr(env, "resample_commands"):
        env.resample_commands = lambda *args, **kwargs: None
    env._post_physics_step_callback = (
        lambda *args, **kwargs: None
    )  # KHÓA CHẾ ĐỘ RESET LỆNH

    env.reset()
    obs_dict = env.get_observations()
    obs_buf, _, obs_history, _ = (
        obs_dict if isinstance(obs_dict, tuple) else (obs_dict, None, None, None)
    )

    control_dt = 1.0 / 50.0
    z_t = None
    last_cmd_v = 0.0
    last_cmd_vy = 0.0
    last_cmd_w = 0.0
    last_yaw_debug = 0.0  # Để tính change trong log

    # --- Chuẩn bị điểm hiển thị vùng ma sát (Friction Zones) ---
    friction_debug_pts = []
    for zone in cfg.FRICTION_ZONES:
        x_min, x_max, y_min, y_max, friction = zone
        if friction != 1.0:
            res = 0.2  # Khoảng cách giữa các quả cầu debug
            # Cạnh trên và dưới
            for x in np.arange(x_min, x_max + res, res):
                friction_debug_pts.append([x, y_min, 0.05])
                friction_debug_pts.append([x, y_max, 0.05])
            # Cạnh trái và phải
            for y in np.arange(y_min, y_max + res, res):
                friction_debug_pts.append([x_min, y, 0.05])
                friction_debug_pts.append([x_max, y, 0.05])

    if friction_debug_pts:
        friction_debug_pts_np = np.array(friction_debug_pts)
    else:
        friction_debug_pts_np = None

    # --- Khởi tạo đo lường hiệu suất (RMS & Jerk Index) ---
    start_time = time.time()
    actual_path_length = 0.0
    last_pos = None
    collision_count = 0
    min_dist_to_obs = float("inf")
    command_history = []

    # Biến tích lũy cho RMS và Jerk
    sum_sq_vx, sum_sq_vy, sum_sq_yaw = 0.0, 0.0, 0.0
    sum_sq_roll, sum_sq_pitch = 0.0, 0.0
    sum_jerk_vx, sum_jerk_vy, sum_jerk_yaw = 0.0, 0.0, 0.0
    prev_delta_vx, prev_delta_vy, prev_delta_yaw = 0.0, 0.0, 0.0
    prev_cmd = np.zeros(3)
    step_count = 0
    in_collision = False
    is_failed = False
    fail_reason = ""
    dist_to_goal = 999.0

    for i in range(1000000):
        t0 = time.perf_counter()
        lidar_pts_world = None
        if hasattr(env.simulator, "lidar_sensor"):
            pts_local, _ = env.simulator.lidar_sensor.read()
            if pts_local.shape[0] > 0:
                base_pos = env.simulator.base_pos[0]
                base_quat = env.simulator.base_quat[0]
                # World frame Lidar points
                pts_world = base_pos + quat_apply(
                    base_quat.repeat(pts_local.shape[0], 1), pts_local
                )
                pts_world_flat = pts_world.reshape(-1, 3)

                # Height filter & Radial Filter (Khôi phục 0.3m theo yêu cầu)
                h_mask = (pts_world_flat[:, 2] > 0.15) & (pts_world_flat[:, 2] < 1.0)
                dists = torch.norm(pts_world_flat[:, :2] - base_pos[:2], dim=-1)
                lidar_pts_world = pts_world_flat[h_mask & (dists > 0.3)]

        if encoder is not None and obs_history is not None:
            with torch.no_grad():
                z_t = encoder(obs_history)

        if cfg.USE_DWA and dwa_solver is not None:
            base_pos = env.simulator.base_pos[0]
            base_quat = env.simulator.base_quat[0]
            euler = get_euler_xyz(base_quat.unsqueeze(0))
            yaw = euler[0, 2].item()

            target_pt = torch.tensor(cfg.GOAL_POS, device=env.device)

            if cfg.USE_SAFETY_FILTER and safety_filter is not None and z_t is not None:
                safety_obs = torch.cat([z_t, last_safety_alpha], dim=-1)
                with torch.no_grad():
                    raw_actions = safety_filter(safety_obs)
                prev_alpha = last_safety_alpha.clone()
                raw_alpha = (torch.tanh(raw_actions) + 1.0) / 2.0
                delta = raw_alpha - prev_alpha
                rate = torch.where(
                    delta > 0, torch.full_like(delta, 0.15), torch.full_like(delta, 0.5)
                )
                alpha_scale = torch.clamp(prev_alpha + rate * delta, 0.0, 1.0)
                last_safety_alpha = alpha_scale.clone()
                # Scale DWA physical limits symmetrically
                v_max = alpha_scale[0, 0].item() * V_MAX_MAX[0].item()
                vy_max = alpha_scale[0, 1].item() * V_MAX_MAX[1].item()
                w_max = alpha_scale[0, 2].item() * V_MAX_MAX[2].item()
                dwa_solver.simu_robot.lim_max_vel = v_max
                dwa_solver.simu_robot.lim_min_vel = (
                    cfg.V_MIN_MAX_BACK
                )  # Backward is fixed
                dwa_solver.simu_robot.lim_max_lat_vel = vy_max
                dwa_solver.simu_robot.lim_min_lat_vel = -vy_max
                dwa_solver.simu_robot.lim_max_ang_vel = w_max
                dwa_solver.simu_robot.lim_min_ang_vel = -w_max
            else:
                dwa_solver.simu_robot.lim_max_vel = V_MAX_MAX[0].item()
                dwa_solver.simu_robot.lim_min_vel = cfg.V_MIN_MAX_BACK
                dwa_solver.simu_robot.lim_max_lat_vel = V_MAX_MAX[1].item()
                dwa_solver.simu_robot.lim_min_lat_vel = -V_MAX_MAX[1].item()
                dwa_solver.simu_robot.lim_max_ang_vel = V_MAX_MAX[2].item()
                dwa_solver.simu_robot.lim_min_ang_vel = -V_MAX_MAX[2].item()

            # 4. Tìm Waypoint từ A* hoặc dùng Goal cuối (Lookahead 4.0m để đạt tốc độ cao)
            target_pt = torch.tensor(cfg.GOAL_POS, device=env.device)
            target_path_segment = []
            if global_path:
                # 1. Tìm điểm trên đường đi gần robot nhất
                curr_pos_np = base_pos[:2].cpu().numpy()
                path_np = np.array(global_path)
                dists_to_all = np.linalg.norm(path_np - curr_pos_np, axis=1)
                closest_idx = np.argmin(dists_to_all)

                # 2. Lookahead: Đi dọc theo đường A* khoảng 2.0m
                lookahead_idx = closest_idx
                accum_dist = 0.0
                while lookahead_idx < len(global_path) - 1 and accum_dist < 2.0:
                    accum_dist += math.hypot(
                        global_path[lookahead_idx + 1][0]
                        - global_path[lookahead_idx][0],
                        global_path[lookahead_idx + 1][1]
                        - global_path[lookahead_idx][1],
                    )
                    lookahead_idx += 1

                target_pt = torch.tensor(global_path[lookahead_idx], device=env.device)
                target_path_segment = global_path[closest_idx : lookahead_idx + 1]

            goal_yaw = math.atan2(
                target_pt[1].item() - base_pos[1].item(),
                target_pt[0].item() - base_pos[0].item(),
            )

            obstacles = []
            if lidar_pts_world is not None and lidar_pts_world.shape[0] > 0:
                obstacles = [
                    Obstacle(p[0].item(), p[1].item(), 0.15) for p in lidar_pts_world
                ]

            if i % 20 == 0:
                print(f"[DWA Debug] Num Obstacles: {len(obstacles)}")

            state_dwa = types.SimpleNamespace(
                x=base_pos[0].item(),
                y=base_pos[1].item(),
                th=yaw,
                u_v=last_cmd_v,
                u_vy=last_cmd_vy,
                u_th=last_cmd_w,
            )
            opt_path, top_paths = dwa_solver.calc_input(
                target_pt[0].item(),
                target_pt[1].item(),
                state_dwa,
                obstacles,
                target_path=target_path_segment,
            )
            last_cmd_v, last_cmd_vy, last_cmd_w = (
                opt_path.u_v,
                opt_path.u_vy,
                opt_path.u_th,
            )

            # Anti-Drift: Bật lại vì đã có quyền kiểm soát lệnh
            angle_error = abs(angle_range_corrector(goal_yaw - yaw))
            if angle_error > 0.4:
                last_cmd_v = min(last_cmd_v, 0.05)

            env.commands[:, 0] = last_cmd_v
            env.commands[:, 1] = last_cmd_vy
            env.commands[:, 2] = last_cmd_w

            # --- DEBUG VISUALIZATION ---
            if cfg.SHOW_VIEWER and i % 2 == 0:
                env.simulator._scene.clear_debug_objects()
                # 1. Điểm đích cuối (Màu Cam)
                env.simulator._scene.draw_debug_spheres(
                    goal_pos_numpy, radius=0.2, color=(1, 0.5, 0, 0.8)
                )
                # 2. Đường đi A* (Màu xanh dương)
                if path_numpy is not None:
                    env.simulator._scene.draw_debug_spheres(
                        path_numpy, radius=0.08, color=(0, 0, 1, 0.4)
                    )
                # 5. Lidar (Màu xanh lá) - Vẽ quả cầu nhỏ cho rõ
                if lidar_pts_world is not None and lidar_pts_world.shape[0] > 0:
                    draw_pts = lidar_pts_world[
                        torch.randint(
                            0,
                            lidar_pts_world.shape[0],
                            (min(100, lidar_pts_world.shape[0]),),
                        )
                    ]
                    env.simulator._scene.draw_debug_spheres(
                        draw_pts.cpu().numpy(), radius=0.05, color=(0, 1, 0, 0.4)
                    )

                # 6. Đường viền vùng ma sát khác 1.0 (Màu Tím)
                if friction_debug_pts_np is not None:
                    env.simulator._scene.draw_debug_spheres(
                        friction_debug_pts_np, radius=0.04, color=(1, 0, 1, 0.6)
                    )

        with torch.no_grad():
            actions = (
                policy(obs_buf, obs_history) if encoder is not None else policy(obs_buf)
            )
        cur_friction = get_friction_at(
            env.simulator.base_pos[0, 0].item(),
            env.simulator.base_pos[0, 1].item(),
            cfg,
        )
        nlinks = env.simulator._robot.n_links
        env.simulator._robot.set_friction_ratio(
            torch.full((1, nlinks), cur_friction, device=env.device),
            torch.arange(nlinks, device=env.device),
        )
        if hasattr(env.simulator, "_gs_terrain"):
            env.simulator._gs_terrain.set_friction(cur_friction)
        env.simulator._friction_values[:] = cur_friction

        obs_buf, _, obs_history, _, _, _, _ = env.step(actions.detach())

        # --- Cập nhật đo lường hiệu suất ---
        base_pos_curr = env.simulator.base_pos[0]
        curr_pos_np = base_pos_curr[:2].cpu().numpy()
        if last_pos is not None:
            actual_path_length += np.linalg.norm(curr_pos_np - last_pos)
        last_pos = curr_pos_np

        if lidar_pts_world is not None and lidar_pts_world.shape[0] > 0:
            # 1. Khoảng cách tâm (để báo cáo Min_Dist)
            dists = torch.norm(lidar_pts_world[:, :2] - base_pos_curr[:2], dim=-1)
            step_min_dist = torch.min(dists).item()
            if step_min_dist < min_dist_to_obs:
                min_dist_to_obs = step_min_dist

            # 2. Kiểm tra va chạm hình hộp (Bounding Box)
            # Chuyển điểm Lidar về hệ tọa độ robot (Local Frame)
            base_quat_curr = env.simulator.base_quat[0]
            rel_pts = lidar_pts_world - base_pos_curr
            local_pts = quat_rotate_inverse(
                base_quat_curr.repeat(rel_pts.shape[0], 1), rel_pts
            )

            # Go2 Box: Dọc (X) < 0.36m, Ngang (Y) < 0.15m
            # Chúng ta chỉ kiểm tra XY vì Z đã được lọc từ trước
            collision_mask = (torch.abs(local_pts[:, 0]) < 0.36) & (
                torch.abs(local_pts[:, 1]) < 0.15
            )
            is_colliding = torch.any(collision_mask).item()

            if is_colliding:
                if not in_collision:
                    collision_count += 1
                    in_collision = True
            else:
                in_collision = False

        curr_cmd = np.array([last_cmd_v, last_cmd_vy, last_cmd_w])
        delta_vx = curr_cmd[0] - prev_cmd[0]
        delta_vy = curr_cmd[1] - prev_cmd[1]
        delta_yaw = curr_cmd[2] - prev_cmd[2]

        # Trích xuất Roll/Pitch để đánh giá độ ổn định
        base_quat_now = env.simulator.base_quat[0:1]  # Giữ dạng (1, 4)
        euler_xyz = get_euler_xyz(base_quat_now)[0]  # Lấy kết quả đầu tiên (3,)
        roll, pitch = euler_xyz[0].item(), euler_xyz[1].item()

        # Kiểm tra điều kiện ngã (Lớn hơn 90 độ = 1.57 rad)
        if abs(roll) > 1.57 or abs(pitch) > 1.57:
            is_failed = True
            fail_reason = "Robot bị ngã"
            break

        # RMS Smoothness (Bình phương sai số)
        sum_sq_vx += delta_vx**2
        sum_sq_vy += delta_vy**2
        sum_sq_yaw += delta_yaw**2
        sum_sq_roll += roll**2
        sum_sq_pitch += pitch**2

        # Jerk Index (Sự biến thiên của gia tốc)
        if step_count > 0:
            sum_jerk_vx += abs(delta_vx - prev_delta_vx)
            sum_jerk_vy += abs(delta_vy - prev_delta_vy)
            sum_jerk_yaw += abs(delta_yaw - prev_delta_yaw)

        prev_delta_vx, prev_delta_vy, prev_delta_yaw = delta_vx, delta_vy, delta_yaw
        prev_cmd = curr_cmd.copy()
        step_count += 1

        command_history.append(
            [time.time() - start_time, last_cmd_v, last_cmd_vy, last_cmd_w, roll, pitch]
        )

        dist_to_goal = np.linalg.norm(curr_pos_np - np.array(cfg.GOAL_POS))
        if dist_to_goal < 0.2:
            print(
                f"\n[Thành công] Robot đã về đích! Khoảng cách cuối: {dist_to_goal:.3f}m"
            )
            break

        elapsed = time.perf_counter() - t0
        if control_dt > elapsed:
            time.sleep(control_dt - elapsed)
        if i % 10 == 0:
            cmd = env.commands[0].cpu().numpy()
            pos = env.simulator.base_pos[0].cpu().numpy()
            z_str = (
                f"zt:[{z_t[0,0]:+4.2f}, {z_t[0,1]:+4.2f}, {z_t[0,2]:+4.2f}]"
                if z_t is not None
                else "zt: N/A"
            )
            a_str = f"alpha:[{last_safety_alpha[0,0]:.2f}, {last_safety_alpha[0,1]:.2f}, {last_safety_alpha[0,2]:.2f}]"
            print(
                f"F: {i:06d} | CMD:[{cmd[0]:+4.2f}, {cmd[1]:+4.2f}, {cmd[2]:+4.2f}] | {z_str} | {a_str} | Fri:{cur_friction:.1f} | POS:[{pos[0]:.1f}, {pos[1]:.1f}]"
            )

    # --- Báo cáo tổng hợp kết quả ---
    astar_path_length = 0.0
    if global_path:
        for i in range(len(global_path) - 1):
            astar_path_length += math.hypot(
                global_path[i + 1][0] - global_path[i][0],
                global_path[i + 1][1] - global_path[i][1],
            )

    sf_status = "Bật" if cfg.USE_SAFETY_FILTER else "Tắt"
    total_time = time.time() - start_time

    # Tính toán kết quả cuối cùng (RMS & Jerk)
    div = max(1, step_count)
    rms_vx = math.sqrt(sum_sq_vx / div)
    rms_vy = math.sqrt(sum_sq_vy / div)
    rms_yaw = math.sqrt(sum_sq_yaw / div)
    jerk_vx, jerk_vy, jerk_yaw = (
        sum_jerk_vx / div,
        sum_jerk_vy / div,
        sum_jerk_yaw / div,
    )
    rms_roll = math.sqrt(sum_sq_roll / div)
    rms_pitch = math.sqrt(sum_sq_pitch / div)

    print("\n" + "=" * 50)
    print("      TỔNG HỢP HIỆU SUẤT ĐIỀU HƯỚNG (DWA NO TERRAIN)")
    print("=" * 50)
    print(f"Safety Filter:       {sf_status}")
    print(f"Về đích:             {'Có' if dist_to_goal < 0.2 else 'Không'}")
    print(f"Tổng thời gian:      {total_time:.2f} s")
    print(f"Số lần va chạm:      {collision_count}")
    print(f"Khoảng cách gần nhất:{min_dist_to_obs:.3f} m")
    print(f"Quãng đường thực tế: {actual_path_length:.2f} m")
    print(f"Quãng đường A*:      {astar_path_length:.2f} m")
    print(f"RMS Smoothness (Vx/Vy/Yaw): {rms_vx:.4f} / {rms_vy:.4f} / {rms_yaw:.4f}")
    print(f"Jerk Index (Vx/Vy/Yaw):     {jerk_vx:.4f} / {jerk_vy:.4f} / {jerk_yaw:.4f}")
    print(f"Body Stability (RMS Roll/Pitch): {rms_roll:.4f} / {rms_pitch:.4f} (rad)")
    print("=" * 50)

    # --- Lưu toàn bộ dữ liệu vào file CSV (Merged) ---
    save_dir = "/home/datvu/LeggedGym-Ex/test_results"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Bóc tách tên Map
    map_name = "N/A"
    if hasattr(cfg, "MAP_YAML_PATH"):
        map_name = os.path.basename(os.path.dirname(cfg.MAP_YAML_PATH))

    # 2. Tìm số thứ tự file chưa tồn tại
    algo_type = "dwa"
    sf_suffix = "_with_SF" if cfg.USE_SAFETY_FILTER else ""
    base_prefix = f"nav_data_{algo_type}{sf_suffix}"
    count = 1
    while os.path.exists(os.path.join(save_dir, f"{base_prefix}#{count}.csv")):
        count += 1
    save_path = os.path.join(save_dir, f"{base_prefix}#{count}.csv")

    # 3. Ghi file với cấu trúc gộp (Dạng cột dọc cho Summary)
    reached_goal = "Có" if dist_to_goal < 0.2 else "Không"
    summary_labels = [
        "Algorithm",
        "Terrain",
        "Map",
        "Safety_Filter",
        "Status",
        "Fail_Reason",
        "Remaining_Dist",
        "Total_Time",
        "Collisions",
        "Min_Dist",
        "Actual_Path",
        "AStar_Path",
        "RMS_Vx",
        "RMS_Vy",
        "RMS_Yaw",
        "Jerk_Vx",
        "Jerk_Vy",
        "Jerk_Yaw",
        "RMS_Roll",
        "RMS_Pitch",
    ]
    summary_values = [
        "DWA",
        "No",
        map_name,
        sf_status,
        "Success" if dist_to_goal < 0.2 else "Fail",
        fail_reason if is_failed else ("Timeout" if not (dist_to_goal < 0.2) else "None"),
        f"{dist_to_goal:.3f}",
        f"{total_time:.2f}",
        collision_count,
        f"{min_dist_to_obs:.3f}",
        f"{actual_path_length:.2f}",
        f"{astar_path_length:.2f}",
        f"{rms_vx:.5f}",
        f"{rms_vy:.5f}",
        f"{rms_yaw:.5f}",
        f"{jerk_vx:.5f}",
        f"{jerk_vy:.5f}",
        f"{jerk_yaw:.5f}",
        f"{rms_roll:.5f}",
        f"{rms_pitch:.5f}",
    ]

    with open(save_path, "w") as f:
        # Ghi Summary theo cột dọc
        f.write("Metric,Value\n")
        for label, value in zip(summary_labels, summary_values):
            f.write(f"{label},{value}\n")

        f.write("\n")  # Dòng trống phân cách
        # Ghi History
        f.write("time,vx,vy,vyaw,roll,pitch\n")
        for row in command_history:
            f.write(
                f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f}\n"
            )

    print(f"Toàn bộ kết quả đã được lưu vào: {save_path}")


if __name__ == "__main__":
    args = get_args()
    if args.task == "None":
        args.task = "go2_mesh_dwa"
    play(args)

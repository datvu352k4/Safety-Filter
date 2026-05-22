import os
import sys
import types
import time
import glob
import importlib.util
import math
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

from legged_gym import *
from legged_gym.envs import *
import heapq
import yaml
import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.mppi_lib import MPPI
from legged_gym.utils.math_utils import quat_rotate_inverse, quat_apply, get_euler_xyz
from legged_gym.simulator.genesis_simulator import GenesisSimulator

# ==============================================================================
# ─── 1. CẤU HÌNH HỆ THỐNG ──────────────────────────────────────────────────────
# ==============================================================================


class Go2MeshConfig:
    # ── Thông tin Robot & Policy ──
    LOG_DIR = "/home/datvu/LeggedGym-Ex/logs/go2_rough_terrain/Apr19_12-55-01_ts_terrain_genesis"
    CHECKPOINT = -1  # -1 = load checkpoint mới nhất

    # ── Vật thể 3D (.obj) (Sẽ được nạp động theo --map) ──
    OBJ_FILES = []

    # ── Cấu hình Robot & Mô phỏng ──
    INIT_POS = [0.0, 0.0, 0.42]
    DEFAULT_FRICTION = 1.0

    # Vùng ma sát & sỏi đá (Sẽ được nạp động theo --map)
    FRICTION_ZONES = []
    ROUGH_ZONES = []

    # ── Điều hướng (MPPI Navigation) ──
    USE_MPPI = True
    GOAL_POS = [0.0, 0.0]
    MAP_YAML_PATH = ""

    # Tham số MPPI
    MPPI_HORIZON = 50
    MPPI_SAMPLES = 800
    MPPI_SIGMAS = torch.tensor([0.3, 0.2, 0.3])
    MPPI_LAMBDA = 0.1

    # Giới hạn vận tốc MPPI (Khi Alpha = 1.0)
    V_MAX_MAX = [2.0, 1.0, 1.5]
    V_MIN_MAX = [-1.5, -1.0, -1.5]

    # ── Safety Filter ──
    USE_SAFETY_FILTER = False
    # Model sẽ tự động được tìm file mới nhất trong thư mục go2_safety_terrain

    # ── Hiển thị ──
    SHOW_VIEWER = False


# ==============================================================================
# ─── 2. HÀM BỔ TRỢ & CẤU TRÚC ĐIỀU HƯỚNG ─────────────────────────────────────
# ==============================================================================


# 2.1 Mô hình chuyển động (Dynamics) cho MPPI
def robot_dynamics(state, action):
    # state: [K, 3] (x, y, yaw), action: [K, 3] (vx, vy, wz)
    dt = 0.05  # Bước thời gian dự đoán
    x, y, yaw = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, wz = action[:, 0], action[:, 1], action[:, 2]

    new_state = torch.zeros_like(state)
    new_state[:, 0] = x + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * dt
    new_state[:, 1] = y + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * dt
    new_state[:, 2] = yaw + wz * dt
    return new_state


# 2.2 Thuật toán A* Planner & Smoother
class AStarPlanner:
    def __init__(self, map_yaml_path, min_radius=0.18, soft_radius=0.65):
        if not os.path.exists(map_yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file bản đồ: {map_yaml_path}")
        with open(map_yaml_path, "r") as f:
            map_info = yaml.safe_load(f)
        self.resolution = map_info["resolution"]
        self.origin = map_info["origin"]
        pgm_path = os.path.join(os.path.dirname(map_yaml_path), map_info["image"])
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        occ = (255.0 - img) / 255.0
        base_grid = (occ > map_info["occupied_thresh"]).astype(np.uint8)

        # Calculate distance to nearest obstacle in meters
        dist_img = cv2.distanceTransform(1 - base_grid, cv2.DIST_L2, 5)
        dist_m = dist_img * self.resolution

        # Binary grid for hard collision (impassable)
        self.grid = (dist_m < min_radius).astype(np.uint8)

        # Cost grid for soft penalty (higher penalty closer to wall)
        self.cost_grid = np.zeros_like(dist_m)
        penalty_mask = (dist_m >= min_radius) & (dist_m < soft_radius)
        # Max penalty near min_radius, smoothly decaying to 0 at soft_radius
        self.cost_grid[penalty_mask] = (
            50.0 * (soft_radius - dist_m[penalty_mask]) / (soft_radius - min_radius)
        )

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
                        step_dist = math.hypot(dx, dy)
                        soft_penalty = self.cost_grid[nxt[1], nxt[0]]
                        cost = step_dist + soft_penalty
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


# 2.3 Hàm tính toán chi phí (Cost Function) cho MPPI
def robot_cost_func(state, action, info):
    # state: [K, 3] (x, y, yaw), action: [K, 3]
    target = info["target_point"]
    lidar_pts = info.get("lidar_pts", None)

    # 1. Chi phí khoảng cách tới Waypoint
    dist_to_goal = torch.norm(state[:, :2] - target, dim=1)
    costs = dist_to_goal * 15.0

    # 2. Chi phí hướng mặt (Heading Cost)
    goal_dir = target - state[:, :2]
    target_yaw = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
    yaw_diff = torch.abs(
        torch.atan2(
            torch.sin(target_yaw - state[:, 2]), torch.cos(target_yaw - state[:, 2])
        )
    )
    costs += yaw_diff * 1.5

    # 3. Chi phí tránh vật cản (Lidar Obstacle Cost) - Nâng cấp mạnh mẽ
    if lidar_pts is not None and lidar_pts.shape[0] > 0:
        pts_2d = lidar_pts[..., :2].reshape(-1, 2)
        pos_2d = state[:, :2].reshape(-1, 2)

        # Lidar points are now pre-subsampled in the main loop before MPPI calls this function
        # to ensure the same obstacle points are used across the entire rollout horizon.

        dist_matrix = torch.cdist(pos_2d, pts_2d)  # [K, N]
        min_dist, _ = torch.min(dist_matrix, dim=1)

        # Sync with AStarPlanner: min_radius=0.18, soft_radius=0.65
        safety_margin = 0.65
        collision_threshold = 0.2

        obstacle_mask = min_dist < safety_margin
        costs[obstacle_mask] += 100.0 * torch.square(
            safety_margin - min_dist[obstacle_mask]
        )

        # Hình phạt bổ sung nếu quá sát vật cản
        too_close_mask = min_dist < collision_threshold
        costs[too_close_mask] += 800.0

    # 4. Chi phí lệnh điều khiển (Control Effort Cost)
    costs += torch.norm(action, dim=1) * 1.0

    return costs


# 2.4 Cập nhật cách nạp mạng Safety Filter bằng rsl_rl chuẩn
def load_safety_filter(checkpoint_path, device):
    from legged_gym.envs.go2.go2_ts_terrain.go2_safety_terrain_config import (
        Go2SafetyTerrainCfgPPO,
    )
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import OnPolicyRunner

    print(f"[Safety] Đang nạp mạng Safety chuẩn từ: {checkpoint_path}")
    safety_train_cfg = Go2SafetyTerrainCfgPPO()
    safety_train_cfg_dict = class_to_dict(safety_train_cfg)
    if "algorithm" in safety_train_cfg_dict:
        safety_train_cfg_dict["algorithm"].pop("encoder_lr", None)
        safety_train_cfg_dict["algorithm"].pop("num_encoder_epochs", None)

    class MockSafetyEnv:
        def __init__(self):
            self.num_obs = 6
            self.num_privileged_obs = 72
            self.num_actions = 3
            self.num_envs = 1
            self.device = device

        def get_observations(self):
            return torch.zeros((self.num_envs, self.num_obs), device=self.device)

        def reset(self):
            pass

    mock_env = MockSafetyEnv()
    safety_runner = OnPolicyRunner(
        env=mock_env,
        train_cfg=safety_train_cfg_dict,
        log_dir=os.path.dirname(checkpoint_path),
        device=device,
    )
    safety_runner.load(checkpoint_path)
    safety_policy = safety_runner.get_inference_policy(device=device)

    return safety_policy


import legged_gym.utils.terrain as terrain_module
from legged_gym.utils import terrain_utils


def _custom_curiculum(self):
    """Tạo địa hình hỗn hợp: Phẳng hoặc Sỏi đá dựa trên tọa độ ROUGH_ZONES."""
    cfg = self.cfg
    num_rows = cfg.num_rows
    num_cols = cfg.num_cols
    env_length = cfg.terrain_length
    env_width = cfg.terrain_width

    for j in range(num_cols):
        for i in range(num_rows):
            t = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=cfg.vertical_scale,
                horizontal_scale=cfg.horizontal_scale,
            )

            # Tính toán tọa độ thế giới dựa trên gốc tọa độ (0,0) tại góc bản đồ
            x_start = i * env_length
            x_end = (i + 1) * env_length
            y_start = j * env_width
            y_end = (j + 1) * env_width

            is_rough = False
            for zone in getattr(cfg, "ROUGH_ZONES", []):
                z_x_min, z_x_max, z_y_min, z_y_max = zone[:4]
                # Kiểm tra giao cắt giữa ô (x_start, x_end) và zone
                if max(x_start, z_x_min) < min(x_end, z_x_max) and max(
                    y_start, z_y_min
                ) < min(y_end, z_y_max):
                    is_rough = True
                    break

            if is_rough:
                # Sỏi đá giống play_terrain.py (Level 8: +/- 0.08m)
                terrain_utils.random_uniform_terrain(
                    t,
                    min_height=-0.07,
                    max_height=0.07,
                    step=0.005,
                    downsampled_scale=0.2,
                    terrain_type=self.type,
                    flat_edge=0.0,
                )
            else:
                # Phẳng hoàn toàn
                terrain_utils.pyramid_sloped_terrain(
                    t,
                    slope=0.0,
                    platform_size=self.platform_size,
                    terrain_type=self.type,
                )

            self.add_terrain_to_map(t, i, j)


# Tiến hành Patching
terrain_module.Terrain.curiculum = _custom_curiculum


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def apply_stability_patches(env, args):
    """Áp dụng các bản vá để robot chạy ổn định trên Mesh terrain."""

    # 1. Reset Root States Patch: Đảm bảo robot reset đúng vị trí spawn
    # Ép origins về 0 để robot không bị văng ra xa (vd: -50m, -50m) so với file mesh
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

    # 2. Terrain Info Patch: Ngăn crash khi robot đứng trên vùng không có heightfield
    def custom_terrain_info(self_sim):
        if self_sim._cfg.terrain.mesh_type == "plane":
            if hasattr(self_sim, "_height_around_feet"):
                self_sim._height_around_feet[:] = 0.0
            if hasattr(self_sim, "_normal_vector_around_feet"):
                self_sim._normal_vector_around_feet[:, 0::3] = 0.0
                self_sim._normal_vector_around_feet[:, 1::3] = 0.0
                self_sim._normal_vector_around_feet[:, 2::3] = -1.0
        # Nếu là trimesh, Genesis sẽ tự tính toán độ cao thực tế của sỏi đá

    # 3. Camera Patch: Vô hiệu hóa việc tự động bám theo của lớp cơ sở LeggedRobot
    # Trong LeggedRobot._pre_sim_step có đoạn ép camera bám robot, ta chặn nó ở đây.
    def dummy_set_cam(self_env, pos=None, lookat=None):
        pass

    env.set_viewer_camera = types.MethodType(dummy_set_cam, env)


# ==============================================================================
# ─── 3. THỰC THI CHƯƠNG TRÌNH ──────────────────────────────────────────────────
# ==============================================================================


def play(args):
    cfg = Go2MeshConfig()

    # --- Cấu hình Map động ---
    map_name = args.map if args.map else "map2"
    map_configs = {
        "map1": {
            "OBJ_FILES": [
                {
                    "path": "test_map/map1/map1.obj",
                    "pos": (0.0, 0.0, 0.0),
                    "euler": (90, 0, -90),
                    "scale": 1.0,
                    "fixed": True,
                }
            ],
            "FRICTION_ZONES": [
                [2.0, 7.5, 5.0, 12.0, 0.1],
                [8.0, 10.0, 11.0, 15.0, 0.2],
                [10.0, 14.0, 6.0, 15.0, 0.2],
            ],
            "ROUGH_ZONES": [[8.0, 10.0, 11.0, 15.0], [10.0, 14.0, 6.0, 15.0]],
            "GOAL_POS": [13.0, 17.0],
            "MAP_YAML_PATH": "slam_map/map1/map1.yaml",
            "INIT_POS": [0.0, 0.0, 0.42],
        },
        "map2": {
            "OBJ_FILES": [
                {
                    "path": "test_map/map2/map2.obj",
                    "pos": (0.0, 0.0, 0.0),
                    "euler": (90, 0, -90),
                    "scale": 1.0,
                    "fixed": True,
                }
            ],
            "FRICTION_ZONES": [[1.4, 7.0, 3.0, 8.8, 0.18], [7.0, 8.8, 4.5, 11.0, 0.18]],
            "ROUGH_ZONES": [[1.4, 4.8, 3.0, 8.8], [7.0, 8.8, 4.5, 11.0]],
            "GOAL_POS": [10.0, 10.0],
            "MAP_YAML_PATH": "slam_map/map2/map2.yaml",
            "INIT_POS": [0.0, 0.0, 0.42],
        },
        "warehouse": {
            "OBJ_FILES": [
                {
                    "path": "test_map/warehouse/warehouse.obj",
                    "pos": (0.0, 0.0, 0.0),
                    "euler": (90, 0, -90),
                    "scale": 1.0,
                    "fixed": True,
                }
            ],
            "FRICTION_ZONES": [[4.0, 9.7, 1.5, 8.0, 0.1], [11.3, 12.8, 4.0, 7.8, 0.05]],
            "ROUGH_ZONES": [],
            "GOAL_POS": [12.5, 7.5],
            "MAP_YAML_PATH": "slam_map/warehouse/warehouse.yaml",
            "INIT_POS": [0.0, 0.0, 0.42],
        },
    }

    if map_name in map_configs:
        m_cfg = map_configs[map_name]
        cfg.OBJ_FILES = m_cfg["OBJ_FILES"]
        cfg.FRICTION_ZONES = m_cfg["FRICTION_ZONES"]
        cfg.ROUGH_ZONES = m_cfg["ROUGH_ZONES"]
        cfg.GOAL_POS = m_cfg["GOAL_POS"]
        cfg.MAP_YAML_PATH = m_cfg["MAP_YAML_PATH"]
        if "INIT_POS" in m_cfg:
            cfg.INIT_POS = m_cfg["INIT_POS"]
        print(f"[Map] Đã nạp cấu hình cho: {map_name}")
    else:
        print(
            f"[Cảnh báo] Không tìm thấy cấu hình cho map '{map_name}', dùng mặc định (map2)"
        )

    # Ghi đè Safety Filter từ tham số dòng lệnh nếu có
    if args.safety_filter:
        cfg.USE_SAFETY_FILTER = True
    elif args.no_safety_filter:
        cfg.USE_SAFETY_FILTER = False

    args.headless = not cfg.SHOW_VIEWER

    # Khởi tạo Genesis
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )

    # 1. Nạp Config và Environment từ thư mục Log
    config_path = os.path.join(cfg.LOG_DIR, "go2_ts_terrain_config.py")
    env_path = os.path.join(cfg.LOG_DIR, "go2_ts_terrain.py")

    if not os.path.exists(config_path) or not os.path.exists(env_path):
        print(f"[Lỗi] Không tìm thấy file config hoặc env trong: {cfg.LOG_DIR}")
        return

    config_mod = import_module_from_path("log_config", config_path)
    env_mod = import_module_from_path("log_env", env_path)

    # Đăng ký task mới với tên 'go2_mesh'
    task_registry.register(
        "go2_mesh",
        env_mod.Go2TSTerrain,
        config_mod.Go2TSTerrainCfg(),
        config_mod.Go2TSTerrainCfgPPO(),
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name="go2_mesh")

    # 2. Ghi đè (Override) cấu hình để phù hợp với việc chạy thử nghiệm
    env_cfg.env.num_envs = 1
    env_cfg.env.debug = False
    env_cfg.env.episode_length_s = 3600.0  # Tăng lên 1 tiếng để chạy thoải mái

    env_cfg.seed = getattr(args, "seed", None)
    if env_cfg.seed is None:
        env_cfg.seed = -1
    else:
        env_cfg.seed = int(env_cfg.seed)

    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.rendered_envs_idx = [0]

    # Bắt buộc mesh_type = plane để Genesis khởi tạo mặt nền phẳng trước,
    # sau đó chúng ta sẽ nạp thêm Mesh .obj vào sau.
    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.num_rows = 20
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_length = 2.0
    env_cfg.terrain.terrain_width = 2.0
    env_cfg.terrain.border_size = 5.0
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.ROUGH_ZONES = cfg.ROUGH_ZONES  # Quan trọng: Chuyển giao vùng sỏi đá
    env_cfg.init_state.pos = cfg.INIT_POS

    # Dải lệnh điều khiển (Tốc độ tối đa)
    env_cfg.commands.ranges.lin_vel_x = [-2.0, 2.0]
    env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.5, 1.5]
    env_cfg.commands.heading_command = False  # Dùng joystick trực tiếp

    # Cấu hình nạp Model
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = os.path.basename(cfg.LOG_DIR)
    train_cfg.runner.checkpoint = cfg.CHECKPOINT

    # 3. Patching GenesisSimulator để nạp file .obj (Cách mới ổn định hơn)
    orig_create_sim = GenesisSimulator._create_sim

    def patched_create_sim(self_sim):
        # Gọi hàm gốc để khởi tạo Scene và Plane
        orig_create_sim(self_sim)

        # self_sim._scene đã được tạo ra ở hàm gốc
        scene_inst = self_sim._scene

        # 3.1 Hook thêm Lidar vào lúc build (vì lúc này mới có robot entity)
        orig_build = scene_inst.build

        def patched_build(*args, **kwargs):
            # Lấy robot trực tiếp từ simulator (đã được add trước khi build)
            robot_ent = getattr(self_sim, "_robot", None)

            if robot_ent is not None:
                print(f" -> Gắn Lidar lên robot: Go2")
                pattern = gs.sensors.SphericalPattern(
                    fov=(360.0, 0.0), n_points=(144, 1)
                )
                # Lưu lại tham chiếu cảm biến cho simulator bằng cách bắt giá trị trả về
                self_sim.lidar_sensor = scene_inst.add_sensor(
                    gs.sensors.Lidar(
                        pattern=pattern,
                        entity_idx=robot_ent.idx,
                        pos_offset=(0.0, 0.0, 0.15),
                        max_range=100.0,
                        min_range=0.1,
                        draw_debug=cfg.SHOW_VIEWER,
                    )
                )
            return orig_build(*args, **kwargs)

        scene_inst.build = patched_build

        print("\n[Mesh Terrain] Đang nạp các file .obj...")
        for obj in cfg.OBJ_FILES:
            p = obj["path"]
            if not os.path.isabs(p):
                p = os.path.join(LEGGED_GYM_ROOT_DIR, p)

            if os.path.exists(p):
                print(f" -> Nạp: {p}")
                entity = scene_inst.add_entity(
                    gs.morphs.Mesh(
                        file=p,
                        pos=obj["pos"],
                        euler=obj["euler"],
                        scale=obj["scale"],
                        fixed=obj["fixed"],
                    )
                )
                # Áp dụng ma sát cho mesh
                entity.set_friction(cfg.DEFAULT_FRICTION)
            else:
                print(f" [!] Bỏ qua (Không tìm thấy): {p}")

        # Sau khi thêm Mesh, Simulator sẽ tự gọi scene.build() trong hàm _create_envs() tiếp theo

    GenesisSimulator._create_sim = patched_create_sim

    # 4. Khởi tạo môi trường
    env, _ = task_registry.make_env(name="go2_mesh", args=args, env_cfg=env_cfg)
    apply_stability_patches(env, args)

    # 4.1 Áp dụng ma sát chuẩn 1.0 cho sàn và robot
    if hasattr(env.simulator, "_gs_terrain"):
        env.simulator._gs_terrain.set_friction(cfg.DEFAULT_FRICTION)

    # Cập nhật friction buffer cho robot (để policy nhận diện đúng)
    env.simulator._friction_values[:] = cfg.DEFAULT_FRICTION

    # Cập nhật vật lý thực tế cho robot links
    nlinks = env.simulator._robot.n_links
    ratios = torch.full((1, nlinks), cfg.DEFAULT_FRICTION, device=env.device)
    env.simulator._robot.set_friction_ratio(
        ratios, torch.arange(nlinks, device=env.device)
    )

    # 5. Khởi tạo thuật toán (PPO Runner) và lấy Policy
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name="go2_mesh", args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 5.1 Khởi tạo MPPI Controller
    mppi_solver = None
    if cfg.USE_MPPI:
        print(f"[MPPI] Khởi tạo bộ giải với đích đến: {cfg.GOAL_POS}")
        mppi_solver = MPPI(
            horizon=cfg.MPPI_HORIZON,
            num_samples=cfg.MPPI_SAMPLES,
            dim_state=3,
            dim_control=3,
            dynamics=robot_dynamics,
            cost_func=robot_cost_func,
            u_min=torch.tensor(cfg.V_MIN_MAX, device=env.device),
            u_max=torch.tensor(cfg.V_MAX_MAX, device=env.device),
            sigmas=cfg.MPPI_SIGMAS.to(env.device),
            lambda_=cfg.MPPI_LAMBDA,
            device=env.device,
            seed=env_cfg.seed if env_cfg.seed >= 0 else np.random.randint(0, 10000),
        )

    # 5.2 Khởi tạo Safety Filter
    safety_log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/go2_safety_terrain")
    checkpoints = glob.glob(os.path.join(safety_log_dir, "model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Không tìm thấy file model trong {safety_log_dir}")
    checkpoints.sort(key=os.path.getmtime)
    resume_path = checkpoints[-1]

    safety_filter = load_safety_filter(resume_path, env.device)
    print(f"\n✅ Đã load Safety Model tự động từ: {resume_path}")

    last_safety_alpha = torch.ones(1, 3, device=env.device)
    V_MAX_MAX = torch.tensor(cfg.V_MAX_MAX, device=env.device)
    V_MIN_MAX = torch.tensor(cfg.V_MIN_MAX, device=env.device)

    # 5. Khởi tạo A* và Lập kế hoạch đầu tiên
    map_yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.MAP_YAML_PATH)
    astar = AStarPlanner(map_yaml_path)

    print(f"\n[A*] Đang lập kế hoạch tới đích: {cfg.GOAL_POS}")
    init_base_pos = cfg.INIT_POS[:2]
    global_path = astar.plan(init_base_pos, cfg.GOAL_POS)
    if global_path:
        global_path = simple_smoother(global_path)
        print(f"[A*] Đã tìm thấy đường đi: {len(global_path)} điểm.")
    else:
        print("[A*] Lỗi: Không tìm thấy đường đi!")

    # 5.2 Lưu dữ liệu mốc để vẽ
    goal_pos_numpy = np.array([cfg.GOAL_POS[0], cfg.GOAL_POS[1], 0.25]).reshape(1, 3)
    path_numpy = (
        np.array([[p[0], p[1], 0.1] for p in global_path]) if global_path else None
    )

    # Lấy history encoder nếu có (cần thiết cho locomotion policy hiện đại)
    encoder = getattr(
        ppo_runner.alg.actor_critic,
        "history_encoder",
        getattr(ppo_runner.alg.actor_critic.actor, "history_encoder", None),
    )

    # 6. Khởi tạo Joystick
    joystick = None
    if args.use_joystick:
        try:
            joystick = Joystick(joystick_type=args.joystick_type)
            print("[Info] Joystick đã sẵn sàng.")
        except Exception as e:
            print(f"[Cảnh báo] Không tìm thấy Joystick: {e}")

    # 6.1 Hàm hỗ trợ lấy ma sát theo tọa độ
    def get_friction_at(x, y, cfg):
        for zone in cfg.FRICTION_ZONES:
            x_min, x_max, y_min, y_max, friction = zone
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return friction
        return cfg.DEFAULT_FRICTION

    def get_terrain_at(x, y, cfg):
        for zone in cfg.ROUGH_ZONES:
            x_min, x_max, y_min, y_max = zone[:4]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return "rough"
        return "flat"

    # 7. Vòng lặp mô phỏng
    print("\n[Điều khiển] Bắt đầu mô phỏng. Sử dụng Joystick để di chuyển robot.")
    print("[Mẹo] Dùng chuột để xoay và di chuyển camera tự do.")

    # Ép camera tự do (không bám theo robot)
    args.follow_robot = False

    # Đảm bảo robot reset về vị trí init
    env.reset()

    # Lấy quan sát đầu tiên
    obs_dict = env.get_observations()
    if isinstance(obs_dict, tuple):
        obs_buf, _, obs_history, _ = obs_dict
    else:
        obs_buf = obs_dict
        obs_history = None

    control_dt = 1.0 / 50.0  # 50Hz
    z_t = None

    # --- Chuẩn bị điểm hiển thị vùng ma sát (Friction Zones) ---
    friction_debug_pts = []
    for zone in cfg.FRICTION_ZONES:
        x_min, x_max, y_min, y_max, friction = zone
        if friction != 1.0:
            res = 0.2
            for x in np.arange(x_min, x_max + res, res):
                friction_debug_pts.append([x, y_min, 0.05])
                friction_debug_pts.append([x, y_max, 0.05])
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
    sum_sq_roll, sum_sq_pitch = 0.0, 0.0  # Độ ổn định thân robot
    sum_jerk_vx, sum_jerk_vy, sum_jerk_yaw = 0.0, 0.0, 0.0
    prev_delta_vx, prev_delta_vy, prev_delta_yaw = 0.0, 0.0, 0.0
    prev_cmd = np.zeros(3)
    step_count = 0
    in_collision = False
    is_failed = False
    fail_reason = ""
    dist_to_goal = 999.0
    current_path_idx = 0  # Tránh đi lùi nếu trượt
    stuck_pos_check = None

    for i in range(1000000):
        t0 = time.perf_counter()

        # 1. Luôn đọc dữ liệu Lidar và hiển thị Debug (ngay cả khi không dùng MPPI)
        lidar_pts_world = None
        if hasattr(env.simulator, "lidar_sensor"):
            pts_local, _ = env.simulator.lidar_sensor.read()
            if pts_local.shape[0] > 0:
                base_pos = env.simulator.base_pos[0]
                # Sử dụng trực tiếp base_quat từ simulator (không hoán vị nữa)
                base_quat = env.simulator.base_quat[0]

                # Chuyển đổi pts_local sang World frame dùng quat simulator (đã chuẩn)
                pts_world = base_pos + quat_apply(
                    base_quat.repeat(pts_local.shape[0], 1), pts_local
                )

                pts_world_flat = pts_world.reshape(-1, 3)

                # Height filter & Radial Filter (0.3m) để tránh quét trúng chân robot
                h_mask = (pts_world_flat[:, 2] > 0.15) & (pts_world_flat[:, 2] < 1.0)
                dists = torch.norm(pts_world_flat[:, :2] - base_pos[:2], dim=-1)
                lidar_pts_world = pts_world_flat[h_mask & (dists > 0.3)]

                if cfg.SHOW_VIEWER and i % 2 == 0:
                    # Làm mới hiển thị vật cản
                    env.simulator._scene.clear_debug_objects()

                    # Vẽ lại điểm đích (màu vàng cam)
                    env.simulator._scene.draw_debug_spheres(
                        goal_pos_numpy, radius=0.2, color=(1, 0.5, 0, 0.8)
                    )

                    # Vẽ đường đi A* (màu xanh dương)
                    if path_numpy is not None:
                        env.simulator._scene.draw_debug_spheres(
                            path_numpy, radius=0.08, color=(0, 0, 1, 0.4)
                        )
                    #                    scene_inst.draw_debug_lines(p_pts, color=(0, 1, 0, 0.3))

                    # 6. Đường viền vùng ma sát khác 1.0 (Màu Tím)
                    if friction_debug_pts_np is not None:
                        env.simulator._scene.draw_debug_spheres(
                            friction_debug_pts_np, radius=0.04, color=(1, 0, 1, 0.6)
                        )

        #                    # 7. Vẽ Bounding Box va chạm (Màu Đỏ) để debug
        #                    box_corners_local = torch.tensor(
        #                        [
        #                            [0.365, 0.19, -0.38],
        #                            [0.365, -0.19, -0.38],
        #                            [-0.365, 0.19, -0.38],
        #                            [-0.365, -0.19, -0.38],
        #                            [0.365, 0.19, 0.05],
        #                            [0.365, -0.19, 0.05],
        #                            [-0.365, 0.19, 0.05],
        #                            [-0.365, -0.19, 0.05],
        #                        ],
        #                        device=env.device,
        #                    )
        #                    base_pos_debug = env.simulator.base_pos[0]
        #                    base_quat_debug = env.simulator.base_quat[0]
        #                    box_corners_world = base_pos_debug + quat_apply(
        #                        base_quat_debug.repeat(8, 1), box_corners_local
        #                    )
        #                    env.simulator._scene.draw_debug_spheres(
        #                        box_corners_world.cpu().numpy(),
        #                        radius=0.04,
        #                        color=(1, 0, 0, 0.9),
        #                    )

        # 2. Tính toán z_t (latent) từ lịch sử quan sát TRƯỚC khi dùng cho Safety/MPPI
        if encoder is not None and obs_history is not None:
            with torch.no_grad():
                z_t = encoder(obs_history)

        # 3. Cập nhật Điều hướng (MPPI hoặc Joystick)
        if cfg.USE_MPPI and mppi_solver is not None:
            # Lấy trạng thái robot hiện tại
            base_pos = env.simulator.base_pos[0]
            base_quat = env.simulator.base_quat[0]

            euler = get_euler_xyz(base_quat.unsqueeze(0))
            yaw = euler[0, 2]
            state_now = torch.tensor(
                [base_pos[0], base_pos[1], yaw.item()], device=env.device
            )

            # --- TÌM WAYPOINT TRÊN PATH BẰNG TÍCH LŨY KHOẢNG CÁCH (Giống DWA) ---
            target_pt = torch.tensor(cfg.GOAL_POS, device=env.device)
            if global_path:
                # 1. Tìm điểm trên đường đi gần robot nhất (Chỉ tìm tiến tới, giới hạn 50 điểm)
                curr_pos_np = base_pos[:2].cpu().numpy()
                path_np = np.array(global_path)
                search_end = min(len(global_path), current_path_idx + 50)
                dists_in_window = np.linalg.norm(
                    path_np[current_path_idx:search_end] - curr_pos_np, axis=1
                )
                current_path_idx = current_path_idx + int(np.argmin(dists_in_window))
                closest_idx = current_path_idx

                # 2. Lookahead: Đi dọc theo đường A* khoảng 1.5m
                lookahead_idx = closest_idx
                accum_dist = 0.0
                while lookahead_idx < len(global_path) - 1 and accum_dist < 1.5:
                    accum_dist += math.hypot(
                        global_path[lookahead_idx + 1][0]
                        - global_path[lookahead_idx][0],
                        global_path[lookahead_idx + 1][1]
                        - global_path[lookahead_idx][1],
                    )
                    lookahead_idx += 1

                target_pt = torch.tensor(global_path[lookahead_idx], device=env.device)

            # --- 8. SAFETY FILTER CALCULATION ---
            if cfg.USE_SAFETY_FILTER and safety_filter is not None and z_t is not None:
                # Input: z_t(3) + last_alpha(3) = 6
                safety_obs = torch.cat([z_t, last_safety_alpha], dim=-1)
                with torch.no_grad():
                    raw_actions = safety_filter(safety_obs)

                # Mapping trực tiếp (Đồng bộ End-to-End smoothing của môi trường RL)
                alpha_scale = torch.clamp(
                    (torch.tanh(raw_actions) + 1.0) / 2.0, 0.2, 1.0
                )
                last_safety_alpha = alpha_scale.clone()

                # Cập nhật giới hạn MPPI dựa trên alpha_scale (Scale cả 2 đầu)
                mppi_solver._u_min[0] = alpha_scale[0, 0] * V_MIN_MAX[0]
                mppi_solver._u_max[0] = alpha_scale[0, 0] * V_MAX_MAX[0]
                mppi_solver._u_min[1] = alpha_scale[0, 1] * V_MIN_MAX[1]
                mppi_solver._u_max[1] = alpha_scale[0, 1] * V_MAX_MAX[1]
                mppi_solver._u_min[2] = alpha_scale[0, 2] * V_MIN_MAX[2]
                mppi_solver._u_max[2] = alpha_scale[0, 2] * V_MAX_MAX[2]
            else:
                # Nếu tắt Safety Filter, dùng giới hạn mặc định
                if mppi_solver is not None:
                    mppi_solver._u_min[:] = V_MIN_MAX
                    mppi_solver._u_max[:] = V_MAX_MAX

            # Thực hiện subsample lidar *MỘT LẦN* trước khi đưa vào solver
            if lidar_pts_world is not None and lidar_pts_world.shape[0] > 144:
                lidar_pts_mppi = lidar_pts_world[:144]
            else:
                lidar_pts_mppi = lidar_pts_world

            # Giải MPPI
            mppi_info = {
                "target_point": target_pt,
                "lidar_pts": lidar_pts_mppi,
            }
            optimal_actions, _ = mppi_solver.forward(state_now, mppi_info)
            mppi_cmd = optimal_actions[0]

            if i % 50 == 0:
                goal_dist = torch.norm(state_now[:2] - target_pt).item()
                num_pts = lidar_pts_world.shape[0] if lidar_pts_world is not None else 0
                a_str = f"Alpha:[{last_safety_alpha[0,0]:.2f}, {last_safety_alpha[0,1]:.2f}, {last_safety_alpha[0,2]:.2f}]"
                print(
                    f"[MPPI-A*] Waypoint: ({target_pt[0]:.1f}, {target_pt[1]:.1f}) | {a_str} | Dist: {goal_dist:.2f}m | Lidar: {num_pts}"
                )

            env.commands[:, 0] = mppi_cmd[0]
            env.commands[:, 1] = mppi_cmd[1]
            env.commands[:, 2] = mppi_cmd[2]

        elif joystick:
            joystick.update()
            vx_max = env.cfg.commands.ranges.lin_vel_x[1]
            vy_max = env.cfg.commands.ranges.lin_vel_y[1]
            yaw_max = env.cfg.commands.ranges.ang_vel_yaw[1]

            env.commands[:, 0] = -joystick.ly * vx_max
            env.commands[:, 1] = -joystick.lx * vy_max
            env.commands[:, 2] = -joystick.rx * yaw_max

        # Chạy Policy
        with torch.no_grad():
            if encoder is not None and obs_history is not None:
                # z_t đã được tính ở trên
                actions = policy(obs_buf, obs_history)
            else:
                actions = policy(obs_buf)

        # Cập nhật Ma sát theo tọa độ thực tế
        robot_pos = env.simulator.base_pos[0].cpu().numpy()
        cur_friction = get_friction_at(robot_pos[0], robot_pos[1], cfg)

        # Cập nhật vật lý cho Robot links và Sàn
        nlinks = env.simulator._robot.n_links
        ratios = torch.full((1, nlinks), cur_friction, device=env.device)
        env.simulator._robot.set_friction_ratio(
            ratios, torch.arange(nlinks, device=env.device)
        )

        if hasattr(env.simulator, "_gs_terrain"):
            env.simulator._gs_terrain.set_friction(cur_friction)

        # Cập nhật buffer Ground Truth để giám sát
        env.simulator._friction_values[:] = cur_friction

        # Bước mô phỏng
        step_result = env.step(actions.detach())
        obs_buf, _, obs_history, _, _, _, _ = step_result

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

            # Go2 Box: Dọc (X) < 0.36m, Ngang (Y) < 0.18m
            collision_mask = (torch.abs(local_pts[:, 0]) < 0.36) & (
                torch.abs(local_pts[:, 1]) < 0.18
            )
            is_colliding = torch.any(collision_mask).item()

            if is_colliding:
                if not in_collision:
                    collision_count += 1
                    in_collision = True
            else:
                in_collision = False
        else:
            in_collision = False

        curr_cmd = env.commands[0].cpu().numpy()
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

        # Kiểm tra mắc kẹt (Stuck Detection)
        if step_count % 200 == 0:
            if stuck_pos_check is not None:
                dist_moved = np.linalg.norm(curr_pos_np - stuck_pos_check)
                if dist_moved < 0.1 and abs(curr_cmd[0]) > 0.1:
                    is_failed = True
                    fail_reason = "Robot bị kẹt (Stuck)"
                    break
            stuck_pos_check = curr_pos_np.copy()

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

        # Tính toán độ cao thân xe tương đối so với bàn chân
        feet_pos_z = env.simulator._feet_pos[0, :, 2]
        rel_height = (base_pos_curr[2] - torch.mean(feet_pos_z)).item()

        cur_terrain = get_terrain_at(curr_pos_np[0], curr_pos_np[1], cfg)
        command_history.append(
            [
                time.time() - start_time,
                curr_cmd[0],
                curr_cmd[1],
                curr_cmd[2],
                roll,
                pitch,
                rel_height,
                curr_pos_np[0],
                curr_pos_np[1],
                cur_friction,
                cur_terrain,
            ]
        )

        dist_to_goal = np.linalg.norm(curr_pos_np - np.array(cfg.GOAL_POS))
        if dist_to_goal < 0.5:
            print(
                f"\n[Thành công] Robot đã về đích! Khoảng cách cuối: {dist_to_goal:.3f}m"
            )
            break

        # (Đã gỡ bỏ phần Camera Follow để bảo đảm camera tự do 100%)

        # Giới hạn FPS để quan sát mượt mà
        elapsed = time.perf_counter() - t0
        if control_dt > elapsed:
            time.sleep(control_dt - elapsed)

        # 6. In thông tin debug (10 bước một lần, xuống dòng để dễ theo dõi)
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

    rms_roll = math.sqrt(sum_sq_roll / div)
    rms_pitch = math.sqrt(sum_sq_pitch / div)

    status_str = (
        "THÀNH CÔNG"
        if dist_to_goal < 0.5
        else ("THẤT BẠI: " + fail_reason if is_failed else "THẤT BẠI: Hết thời gian")
    )

    print("\n" + "=" * 50)
    print("      TỔNG HỢP HIỆU SUẤT ĐIỀU HƯỚNG (MPPI TERRAIN)")
    print("=" * 50)
    print(f"Trạng thái:          {status_str}")
    print(f"Khoảng cách còn lại: {dist_to_goal:.3f} m")
    print(f"Safety Filter:       {sf_status}")
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
    # 1. Bóc tách tên Map và tạo thư mục lưu chuyên biệt
    map_name = args.map if args.map else "N/A"
    algo_slug = "mppi"
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "test_results", map_name, algo_slug)
    os.makedirs(save_dir, exist_ok=True)

    # 2. Tìm số thứ tự file chưa tồn tại
    sf_suffix = "_with_SF" if cfg.USE_SAFETY_FILTER else ""
    base_prefix = f"nav_data_{algo_slug}{sf_suffix}"
    count = 1
    while os.path.exists(os.path.join(save_dir, f"{base_prefix}#{count}.csv")):
        count += 1
    save_path = os.path.join(save_dir, f"{base_prefix}#{count}.csv")

    # 3. Ghi file với cấu trúc gộp (Dạng cột dọc cho Summary)
    reached_goal = "Có" if dist_to_goal < 0.5 else "Không"
    summary_labels = [
        "Algorithm",
        "Terrain",
        "Map",
        "Safety_Filter",
        "Seed",
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
        "MPPI",
        "Terrain",
        map_name,
        sf_status,
        str(env_cfg.seed),
        "Success" if dist_to_goal < 0.5 else "Fail",
        (
            fail_reason
            if is_failed
            else ("Timeout" if not (dist_to_goal < 0.5) else "None")
        ),
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
        f.write("time,vx,vy,vyaw,roll,pitch,rel_height,x,y,friction,terrain\n")
        for row in command_history:
            f.write(
                f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f},{row[7]:.4f},{row[8]:.4f},{row[9]:.2f},{row[10]}\n"
            )

    print(f"Toàn bộ kết quả đã được lưu vào: {save_path}")


if __name__ == "__main__":
    args = get_args()
    # Mặc định dùng task go2_mesh nếu người dùng không truyền vào
    if args.task == "None":
        args.task = "go2_mesh"
    play(args)

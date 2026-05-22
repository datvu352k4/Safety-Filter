import os
import sys
import types
import time
import importlib.util
import math
import numpy as np
import torch
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
    # Đường dẫn đến thư mục log chứa config và model
    LOG_DIR = "/home/datvu/LeggedGym-Ex/logs/go2_rough_terrain/Apr10_16-52-15_ts_terrain_genesis"
    CHECKPOINT = 1500  # Số iteration của model muốn load (model_1500.pt)

    # ── Vật thể 3D (.obj) nạp vào Scene ──
    # Bạn có thể thêm nhiều file obj vào danh sách này
    OBJ_FILES = [
        {
            "path": "test_map/map1/map1.obj",  # Đường dẫn (tương đối so với LEGGED_GYM_ROOT_DIR hoặc tuyệt đối)
            "pos": (0.0, 0.0, 0.0),  # Vị trí (x, y, z)
            "euler": (90, 0, -90),  # Góc xoay (roll, pitch, yaw) tính theo độ
            "scale": 1.0,  # Tỉ lệ phóng to/thu nhỏ
            "fixed": True,  # Cố định vật thể (không bị rơi/di chuyển)
        },
    ]

    # ── Cấu hình Robot ban đầu ──
    INIT_POS = [0.0, 0.0, 0.42]  # Vị trí xuất phát của robot [x, y, z]

    # ── Ma sát (Friction) ──
    DEFAULT_FRICTION = 1.0
    SLIPPERY_FRICTION = 0.1
    # Định nghĩa vùng trơn: [x_min, x_max, y_min, y_max]
    SLIPPERY_ZONE = [3.0, 11.0, 6.0, 11.0]

    # ── Điều hướng (MPPI Navigation) ──
    USE_MPPI = True
    GOAL_POS = [13.0, 15.0]  # Tọa độ đích [x, y]

    # Tham số MPPI
    MPPI_HORIZON = 40  # Số bước dự đoán
    MPPI_SAMPLES = 800  # Số lượng mẫu thử quỹ đạo
    MPPI_SIGMAS = torch.tensor(
        [0.15, 0.1, 0.2]
    )  # Giảm 50% để quỹ đạo mượt hơn, bớt rung
    MPPI_LAMBDA = 0.1  # Tăng từ 0.05 để pha trộn các mẫu mượt hơn


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
    def __init__(self, map_yaml_path, inflation_radius=0.5):
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

        # 1000 điểm lidar có thể làm chậm cdist, chúng ta subsample nếu cần
        if pts_2d.shape[0] > 100:
            pts_2d = pts_2d[torch.randint(0, pts_2d.shape[0], (100,))]

        dist_matrix = torch.cdist(pos_2d, pts_2d)  # [K, N]
        min_dist, _ = torch.min(dist_matrix, dim=1)

        # Hàm phạt: Giảm khoảng cách an toàn để robot bớt "nhát"
        safety_margin = 0.6  # Giảm từ 1.0 xuống 0.6
        collision_threshold = 0.35  # Giảm nhẹ ngưỡng va chạm

        obstacle_mask = min_dist < safety_margin
        costs[obstacle_mask] += 2000.0 * torch.square(
            safety_margin - min_dist[obstacle_mask]
        )

        # Hình phạt bổ sung "Vạn lùi" nếu quá sát vật cản
        too_close_mask = min_dist < collision_threshold
        costs[too_close_mask] += 5000.0

    # 4. Chi phí lệnh điều khiển (Control Effort Cost) - Tăng để robot di chuyển điềm đạm hơn
    costs += torch.norm(action, dim=1) * 1.5

    return costs


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

    env.simulator._calc_terrain_info_around_feet = types.MethodType(
        custom_terrain_info, env.simulator
    )

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

    # Khởi tạo Genesis
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

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
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.rendered_envs_idx = [0]

    # Bắt buộc mesh_type = plane để Genesis khởi tạo mặt nền phẳng trước,
    # sau đó chúng ta sẽ nạp thêm Mesh .obj vào sau.
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False

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
                        draw_debug=True,
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
            u_min=torch.tensor([-1.0, -1.0, -1.5], device=env.device),
            u_max=torch.tensor([2.0, 1.0, 1.5], device=env.device),
            sigmas=cfg.MPPI_SIGMAS.to(env.device),
            lambda_=cfg.MPPI_LAMBDA,
            device=env.device,
        )

    # 5. Khởi tạo A* và Lập kế hoạch đầu tiên
    map_yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, "slam_map", "map1", "map1.yaml")
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
    def get_friction_at(x, y):
        x1, x2, y1, y2 = cfg.SLIPPERY_ZONE
        if x1 <= x <= x2 and y1 <= y <= y2:
            return cfg.SLIPPERY_FRICTION
        return cfg.DEFAULT_FRICTION

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
                z_mask = (pts_world_flat[:, 2] > 0.15) & (pts_world_flat[:, 2] < 1.0)
                lidar_pts_world = pts_world_flat[z_mask]

                if i % 2 == 0:
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

                    # Vẽ các điểm Lidar (màu xanh lá)
                    if lidar_pts_world is not None and lidar_pts_world.shape[0] > 0:
                        draw_pts = lidar_pts_world[
                            torch.randint(
                                0,
                                lidar_pts_world.shape[0],
                                (min(50, lidar_pts_world.shape[0]),),
                            )
                        ]
                        env.simulator._scene.draw_debug_spheres(
                            draw_pts.cpu().numpy(), radius=0.05, color=(0, 1, 0, 0.6)
                        )

        # 2. Cập nhật Điều hướng (MPPI hoặc Joystick)
        if cfg.USE_MPPI and mppi_solver is not None:
            # Lấy trạng thái robot hiện tại
            base_pos = env.simulator.base_pos[0]
            base_quat = env.simulator.base_quat[0]

            euler = get_euler_xyz(base_quat.unsqueeze(0))
            yaw = euler[0, 2]
            state_now = torch.tensor(
                [base_pos[0], base_pos[1], yaw.item()], device=env.device
            )

            # --- TÌM WAYPOINT TRÊN PATH ---
            # Mặc định là đích cuối
            target_pt = torch.tensor(cfg.GOAL_POS, device=env.device)
            if global_path:
                # Tìm điểm xa nhất trên path trong tầm nhìn (vd: 1.5m) để MPPI hướng tới
                for p in reversed(global_path):
                    dist = math.hypot(
                        p[0] - state_now[0].item(), p[1] - state_now[1].item()
                    )
                    if dist < 1.5:
                        target_pt = torch.tensor(p, device=env.device)
                        break

            # Giải MPPI
            mppi_info = {
                "target_point": target_pt,
                "lidar_pts": lidar_pts_world,
            }
            optimal_actions, _ = mppi_solver.forward(state_now, mppi_info)
            mppi_cmd = optimal_actions[0]

            if i % 50 == 0:
                goal_dist = torch.norm(state_now[:2] - target_pt).item()
                num_pts = lidar_pts_world.shape[0] if lidar_pts_world is not None else 0
                print(
                    f"[MPPI-A*] Waypoint: ({target_pt[0]:.1f}, {target_pt[1]:.1f}) | Dist: {goal_dist:.2f}m | Lidar: {num_pts}"
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
            z_t = None
            if encoder is not None and obs_history is not None:
                # Tính toán z_t (latent) từ lịch sử quan sát
                z_t = encoder(obs_history)
                actions = policy(obs_buf, obs_history)
            else:
                actions = policy(obs_buf)

        # Cập nhật Ma sát theo tọa độ thực tế
        robot_pos = env.simulator.base_pos[0].cpu().numpy()
        cur_friction = get_friction_at(robot_pos[0], robot_pos[1])

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

        # (Đã gỡ bỏ phần Camera Follow để bảo đảm camera tự do 100%)

        # Giới hạn FPS để quan sát mượt mà
        elapsed = time.perf_counter() - t0
        if control_dt > elapsed:
            time.sleep(control_dt - elapsed)

        # In thông tin debug mỗi 10 frame
        if i % 10 == 0:
            cmd = env.commands[0].cpu().numpy()
            pos = env.simulator.base_pos[0].cpu().numpy()
            # In thêm z_t và Ma sát thực tế
            z_str = (
                f"z_t:[{z_t[0,0]:+4.2f}, {z_t[0,1]:+4.2f}, {z_t[0,2]:+4.2f}]"
                if z_t is not None
                else "z_t: N/A"
            )
            print(
                f"Frame: {i:06d} | CMD: [x:{cmd[0]:+4.2f}, y:{cmd[1]:+4.2f}, yaw:{cmd[2]:+4.2f}] | {z_str} | Fri:{cur_friction:.1f} | POS: [x:{pos[0]:4.1f}, y:{pos[1]:4.1f}, z:{pos[2]:4.2f}]",
                end="\r",
            )


if __name__ == "__main__":
    args = get_args()
    # Mặc định dùng task go2_mesh nếu người dùng không truyền vào
    if args.task == "None":
        args.task = "go2_mesh"
    play(args)

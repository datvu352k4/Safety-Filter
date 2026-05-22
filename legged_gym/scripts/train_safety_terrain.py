import os
import torch
from legged_gym import SIMULATOR, LEGGED_GYM_ROOT_DIR
import genesis as gs
from legged_gym.utils import get_args, task_registry, class_to_dict

from legged_gym.envs.go2.go2_ts_terrain.go2_safety_terrain_config import (
    Go2SafetyTerrainCfg,
    Go2SafetyTerrainCfgPPO,
)
from legged_gym.envs.go2.go2_ts_terrain.go2_safety_terrain_env import (
    Go2SafetyTerrainEnvWrapper,
)


def train_safety_filter(args):
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name="go2_ts_terrain")

    # Khoá resampling của base env: safety filter tự điều khiển command
    base_env_cfg.commands.resampling_time = 100000.0

    safety_env_cfg = Go2SafetyTerrainCfg()
    safety_train_cfg = Go2SafetyTerrainCfgPPO()

    base_env, _ = task_registry.make_env(
        name="go2_ts_terrain", args=args, env_cfg=base_env_cfg
    )

    base_train_cfg.runner.resume = True
    base_train_cfg.runner.load_run = "Apr19_12-55-01_ts_terrain_genesis"

    old_runner, _ = task_registry.make_alg_runner(
        env=base_env, name="go2_ts_terrain", args=args, train_cfg=base_train_cfg
    )

    ll_policy = old_runner.get_inference_policy(device=base_env.device)
    ll_model = old_runner.alg.actor_critic
    ll_model.eval()

    safety_env = Go2SafetyTerrainEnvWrapper(
        base_env=base_env,
        ll_policy=ll_policy,
        ll_model=ll_model,
        cfg=safety_env_cfg,
    )

    from rsl_rl.runners import OnPolicyRunner

    safety_train_cfg_dict = class_to_dict(safety_train_cfg)
    safety_train_cfg_dict["algorithm"].pop("encoder_lr", None)
    safety_train_cfg_dict["algorithm"].pop("num_encoder_epochs", None)

    log_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", safety_train_cfg.runner.experiment_name
    )

    safety_runner = OnPolicyRunner(
        env=safety_env,
        train_cfg=safety_train_cfg_dict,
        log_dir=log_dir,
        device=base_env.device,
    )

    print(">>> BẮT ĐẦU HUẤN LUYỆN SAFETY FILTER TERRAIN <<<")
    print(f"    actor obs dim : {safety_env_cfg.env.num_observations}")
    print(f"    critic obs dim: {safety_env_cfg.env.num_privileged_obs}")
    print(f"    friction range: {safety_env_cfg.domain_rand.friction_range}")

    safety_runner.learn(
        num_learning_iterations=safety_train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train_safety_filter(args)

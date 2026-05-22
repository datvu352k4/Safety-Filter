import os
import torch
from legged_gym import SIMULATOR, LEGGED_GYM_ROOT_DIR
import genesis as gs
from legged_gym.utils import get_args, task_registry, class_to_dict

from legged_gym.envs.go2.go2_ts.go2_safety_config import Go2SafetyCfg, Go2SafetyCfgPPO
from legged_gym.envs.go2.go2_ts.go2_safety_env import Go2SafetyEnvWrapper


def train_safety_filter(args):
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name="go2_ts")

    # FIX LỖI CHÍ MẠNG: Khóa chức năng Base Env tự động đổi lệnh
    base_env_cfg.commands.resampling_time = 100000.0

    safety_env_cfg = Go2SafetyCfg()
    safety_train_cfg = Go2SafetyCfgPPO()

    base_env, _ = task_registry.make_env(name="go2_ts", args=args, env_cfg=base_env_cfg)

    base_train_cfg.runner.resume = True
    base_train_cfg.runner.load_run = "Mar23_20-44-57_ts_genesis"

    old_runner, _ = task_registry.make_alg_runner(
        env=base_env, name="go2_ts", args=args, train_cfg=base_train_cfg
    )

    ll_policy = old_runner.get_inference_policy(device=base_env.device)
    ll_model = old_runner.alg.actor_critic
    ll_model.eval()

    safety_env = Go2SafetyEnvWrapper(
        base_env=base_env,
        ll_policy=ll_policy,
        ll_model=ll_model,
        cfg=safety_env_cfg,
    )

    from rsl_rl.runners import OnPolicyRunner

    safety_train_cfg_dict = class_to_dict(safety_train_cfg)
    safety_train_cfg_dict["algorithm"].pop("encoder_lr", None)
    safety_train_cfg_dict["algorithm"].pop("num_encoder_epochs", None)

    safety_runner = OnPolicyRunner(
        env=safety_env,
        train_cfg=safety_train_cfg_dict,
        log_dir=os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", safety_train_cfg.runner.experiment_name
        ),
        device=base_env.device,
    )

    print(">>> BẮT ĐẦU HUẤN LUYỆN BỘ LỌC AN TOÀN (SAFETY FILTER) <<<")
    safety_runner.learn(
        num_learning_iterations=safety_train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train_safety_filter(args)

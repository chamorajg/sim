from dataclasses import dataclass


@dataclass
class TDMPCConfigs:
    seed: int = 42
    task: str = "walk"
    exp_name: str = "dora"
    device: str = "cuda:0"
    num_envs: int = 10
    max_clip_actions: float = 18.0
    clip_actions: str = max_clip_actions
    episode_length: int = 100
    max_episode_length: int = 1000

    lr: float = 1e-3
    modality: str = "state"
    enc_dim: int = 512
    mlp_dim = [512, 256]
    dynamics_mlp_dim = [512, 256]
    reward_mlp_dim = [512, 256]
    latent_dim: int = 100

    iterations: int = 12
    num_samples: int = 512
    num_elites: int = 50
    mixture_coef: float = 0.05
    min_std: float = 2.0
    temperature: float = 0.5
    momentum: float = 0.1
    horizon: int = 5
    std_schedule: str = f"linear(5.0, {min_std}, {5 * max_episode_length})"
    horizon_schedule: str = f"linear(1, {horizon},  {5 * max_episode_length})"

    batch_size: int = 8192
    max_buffer_size: int = int(5e6)
    reward_coef: float = 1
    value_coef: float = 0.75
    consistency_coef: float = 2
    rho: float = 0.75
    kappa: float = 0.1
    per_alpha: float = 0.6
    per_beta: float = 0.4
    grad_clip_norm: float = 50
    seed_steps: int = 1200
    update_freq: int = 3
    tau: int = 0.05

    discount: float = 0.99
    buffer_device: str = "cpu"
    train_steps: int = int(1e6)
    num_q: int = 2

    action_repeat: int = 2

    save_model: bool = True
    save_video: bool = False
    save_buffer: bool = False
    eval_model: bool = False
    eval_freq_episode: int = 10
    eval_episodes: int = 1
    save_buffer_freq_episode: int = 50
    save_model_freq_episode: int = 10

    use_wandb: bool = False
    wandb_entity: str = "crajagopalan"
    wandb_project: str = "xbot"


@dataclass
class EvalTDMPCConfigs(TDMPCConfigs):
    seed: int = 42
    task: str = "walk"
    exp_name: str = "dora"
    device: str = "cuda:0"
    horizon: int = 5
    min_std: float = 2.0
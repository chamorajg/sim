"""Trains a humanoid to stand up."""

import argparse
import isaacgym
import os
import cv2
import torch
import imageio
from sim.envs import task_registry
from sim.utils.helpers import get_args
from sim.algo.tdmpc.src import logger
from sim.algo.tdmpc.src.algorithm.tdmpc import TDMPC
from sim.algo.tdmpc.src.config import EvalTDMPCConfigs
from isaacgym import gymapi
import time
import numpy as np
from pathlib import Path
import random
from datetime import datetime
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, render_size=384, fps=25):
        self.save_dir = (root_dir / "eval_video") if root_dir else None
        self.render_size = render_size
        self.fps = fps
        self.enabled = False
        logger.make_dir(self.save_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Creates a directory to store videos.
        self.dir = os.path.join(self.save_dir, now + ".gif")
        self.frames = []

    def init(self, env, h1, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and enabled
        self.record(env, h1)

    def record(self, env, h1, reward: float = None):
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (480, 360))
        # Overlay reward if provided
        if reward is not None:
            cv2.putText(
                img,
                f"Reward: {reward:.2f}",
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        self.frames.append(img)

    def save(
        self,
    ):
        imageio.mimsave(
                self.dir,
                self.frames,
                format='GIF',
                fps=self.fps,
                loop=0  # 0 for infinite loop
            )


def evaluate(test_env, agent, h1, step, video, action_repeat=1):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    state, _ = test_env.reset()
    dones, ep_reward, t = torch.tensor([False] * test_env.num_envs), torch.tensor([0.0] * test_env.num_envs), 0
    if video:
        video.init(test_env, h1, enabled=True)
    for i in tqdm(range(int(500 // action_repeat))):
        actions = agent.plan(state, eval_mode=True, step=step, t0=t == 0)
        for _ in range(action_repeat):
            state, _, rewards, dones, infos = test_env.step(actions)
            ep_reward += rewards.cpu()
            t += 1
            if video:
                video.record(test_env, h1, rewards.mean().item())
    episode_rewards.append(ep_reward)
    if video:
        video.save()
    print(f"Timestep : {t} Episode Rewards - {torch.cat(episode_rewards).mean().item()}")
    return torch.nanmean(torch.cat(episode_rewards)).item()


def play(args: argparse.Namespace) -> None:
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args)

    fp = "/home/kasm-user/sim/logs/2024-10-11_00-30-21_walk_state_dora/models/tdmpc_policy_1010.pt"
    config = torch.load(fp, weights_only=True)["config"]
    tdmpc_cfg = EvalTDMPCConfigs()
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(3, -3, 1)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
    )

    set_seed(tdmpc_cfg.seed)
    work_dir = (
        Path().cwd() / __LOGS__ / f"{tdmpc_cfg.task}_{tdmpc_cfg.modality}_{tdmpc_cfg.exp_name}_{str(tdmpc_cfg.seed)}"
    )

    state, _ = env.reset()

    tdmpc_cfg.obs_shape = state.shape[1:]
    tdmpc_cfg.action_shape = env.num_actions
    tdmpc_cfg.action_dim = env.num_actions
    tdmpc_cfg.num_envs = env.num_envs

    L = logger.Logger(work_dir, tdmpc_cfg)
    log_dir = logger.make_dir(work_dir)
    video = VideoRecorder(log_dir)

    agent = TDMPC(tdmpc_cfg)

    agent.load(fp)
    step = 0
    episode_idx, start_time = 0, time.time()
    if fp is not None:
        episode_idx = int(fp.split(".")[0].split("_")[-1])
        step = episode_idx * tdmpc_cfg.episode_length

    # Log training episode
    evaluate(env, agent, h1, step, video, tdmpc_cfg.action_repeat)
    print("Testing completed successfully")


if __name__ == "__main__":
    play(get_args())

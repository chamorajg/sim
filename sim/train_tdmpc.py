"""Trains a humanoid to stand up."""

import argparse
import isaacgym
import torch
from sim.envs import task_registry
from sim.utils.helpers import get_args
from sim.algo.tdmpc.src import logger
from sim.algo.tdmpc.src.algorithm.helper import Episode, ReplayBuffer
from sim.algo.tdmpc.src.algorithm.tdmpc import TDMPC
from sim.algo.tdmpc.src.config import TDMPCConfigs
from datetime import datetime
from isaacgym import gymapi
from typing import List
import time
import numpy as np
from pathlib import Path
import random

torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(test_env, agent, h1, num_episodes, step, env_step, video, action_repeat=1):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, privileged_obs = test_env.reset()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]
        dones, ep_reward, t = torch.tensor([False]), 0, 0
        if video:
            video.init(test_env, h1, enabled=(i == 0))
        while not dones[0].item():
            actions = agent.plan(state, eval_mode=True, step=step, t0=t == 0)
            for _ in range(action_repeat):
                obs, privileged_obs, rewards, dones, infos = test_env.step(actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]
                ep_reward += rewards[0]
                if video:
                    video.record(test_env, h1)
            t += 1
        episode_rewards.append(ep_reward)
        if video:
            video.save(env_step)
    return torch.nanmean(torch.tensor(episode_rewards)).item()


def train(args: argparse.Namespace) -> None:
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args)

    tdmpc_cfg = TDMPCConfigs()

    if tdmpc_cfg.save_video:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 160
        camera_properties.height = 120
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(3, -3, 1)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
        )

    set_seed(tdmpc_cfg.seed)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = Path().cwd() / __LOGS__ / f"{now}_{tdmpc_cfg.task}_{tdmpc_cfg.modality}_{tdmpc_cfg.exp_name}"

    state, _ = env.reset()

    print(f"Environment max episode length : {env.max_episode_length}")
    tdmpc_cfg.obs_shape = state.shape[1:]
    tdmpc_cfg.action_shape = env.num_actions
    tdmpc_cfg.action_dim = env.num_actions
    tdmpc_cfg.num_envs = env.num_envs
    tdmpc_cfg.max_episode_length = int(env.max_episode_length)
    tdmpc_cfg.episode_capacity = int(env.max_episode_length) // tdmpc_cfg.action_repeat
    tdmpc_cfg.max_clip_actions = env.cfg.normalization.clip_actions
    tdmpc_cfg.clip_actions = f"{env.cfg.normalization.clip_actions}"

    agent = TDMPC(tdmpc_cfg)
    buffer = ReplayBuffer(tdmpc_cfg)
    fp = None

    init_step, env_step = 0, 0
    episode_idx, start_time = 0, time.time()

    if fp is not None:
        init_step, env_step = agent.load(fp)
        if init_step is None:
            episode_idx = int(fp.split(".")[0].split("_")[-1])
            init_step = episode_idx * tdmpc_cfg.episode_length
    env_step = init_step * tdmpc_cfg.episode_length
    episode = Episode(tdmpc_cfg, state)

    L = logger.Logger(work_dir, tdmpc_cfg)

    for step in range(init_step, tdmpc_cfg.train_steps + tdmpc_cfg.episode_length, tdmpc_cfg.episode_length):
        if tdmpc_cfg.init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf, high=int(env.max_episode_length - (tdmpc_cfg.horizon * tdmpc_cfg.action_repeat))
            )
        if episode.full:
            episode = Episode(tdmpc_cfg, state)
        for i in range(tdmpc_cfg.episode_length):
            actions = agent.plan(state, t0=episode.first, eval_mode=False, step=step)
            current_state = state.clone()
            total_rewards, total_dones, total_timeouts = [], [], []
            for _ in range(tdmpc_cfg.action_repeat):
                state, privileged_obs, rewards, dones, infos = env.step(actions)
                next_state = state.clone()
                total_rewards.append(rewards)
                total_dones.append(dones)
                total_timeouts.append(infos["time_outs"])
            episode += (
                current_state,
                next_state,
                actions,
                torch.stack(total_rewards).sum(dim=0),
                torch.stack(total_dones).any(dim=0),
                torch.stack(total_timeouts).any(dim=0),
                torch.stack(total_dones).any(dim=0),
            )
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= tdmpc_cfg.seed_steps:
            num_updates = tdmpc_cfg.seed_steps if step == tdmpc_cfg.seed_steps else tdmpc_cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step += int(tdmpc_cfg.episode_length * tdmpc_cfg.num_envs)
        common_metrics = {
            "episode": episode_idx,
            "step": step + tdmpc_cfg.episode_length,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward.sum().item() / env.num_envs,
            "mean_episode_length": episode.episode_length,
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category="train")

        # Evaluate agent periodically
        if tdmpc_cfg.save_model and episode_idx % tdmpc_cfg.save_model_freq_episode == 0:
            L.save(
                agent,
                f"tdmpc_policy_{int(step // tdmpc_cfg.episode_length) + 1}.pt",
                step + tdmpc_cfg.episode_length,
                env_step,
            )
        if tdmpc_cfg.save_buffer and episode_idx % tdmpc_cfg.save_buffer_freq_episode == 0:
            buffer.save(str(work_dir / "buffer.pt"))
        if tdmpc_cfg.eval_model and episode_idx % tdmpc_cfg.eval_freq_episode == 0:
            common_metrics["episode_reward"] = evaluate(
                env,
                agent,
                h1 if L.video is not None else None,
                tdmpc_cfg.eval_episodes,
                step,
                env_step,
                L.video,
                tdmpc_cfg.action_repeat,
            )
            L.log(common_metrics, category="eval")

    print("Training completed successfully")


if __name__ == "__main__":
    # python -m sim.humanoid_gym.train
    train(get_args())

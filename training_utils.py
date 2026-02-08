"""
Shared training utilities for Trace-based agent training scripts.
"""
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import opto.trace as trace


def create_experiment_dir(game_name, timestamp, base_dir="logs"):
    """
    Create a per-experiment directory with subdirs for checkpoints and GIFs.

    Args:
        game_name: Short game identifier (e.g. "breakout", "pong")
        timestamp: Timestamp string for uniqueness (e.g. "20260207_123456")
        base_dir: Parent directory (default "logs")

    Returns:
        dict with keys: root, log_file, perf_csv, trace_ckpt_dir, gif_dir
    """
    root = Path(base_dir) / f"{game_name}_{timestamp}"
    trace_ckpt_dir = root / "trace_ckpt"
    gif_dir = root / "gifs"

    root.mkdir(parents=True, exist_ok=True)
    trace_ckpt_dir.mkdir(exist_ok=True)
    gif_dir.mkdir(exist_ok=True)

    return {
        "root": root,
        "log_file": root / "train.log",
        "perf_csv": root / "perf.csv",
        "trace_ckpt_dir": trace_ckpt_dir,
        "gif_dir": gif_dir,
    }


def rollout(env, horizon, policy):
    """
    Rollout a policy in an environment for a specified number of steps.
    
    Args:
        env: The environment to run the policy in
        horizon: Maximum number of steps to run
        policy: The policy to execute
        
    Returns:
        tuple: (trajectory dict, error or None)
    """
    try:
        obs, _ = env.reset()
        trajectory = dict(
            observations=[], 
            actions=[], 
            rewards=[], 
            terminations=[], 
            truncations=[], 
            infos=[], 
            steps=0
        )
        trajectory["observations"].append(obs)
        
        for _ in range(horizon):
            error = None
            try:
                action = policy(obs)
                next_obs, reward, termination, truncation, info = env.step(action)
            except trace.ExecutionError as e:
                error = e
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            
            if error is None:
                trajectory["observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminations"].append(termination)
                trajectory["truncations"].append(truncation)
                trajectory["infos"].append(info)
                trajectory["steps"] += 1
                if termination or truncation:
                    break
                obs = next_obs
    finally:
        env.close()
    
    return trajectory, error


def evaluate_policy(policy, env_class, env_name, num_episodes=10, steps_per_episode=4000,
                frameskip=1, repeat_action_probability=0.0, logger=None,
                gif_path=None, gif_fps=30):
    """
    Evaluate a policy over multiple episodes and return statistics.

    Args:
        policy: The policy to evaluate
        env_class: The environment class to instantiate
        env_name: Environment name to pass to env_class
        num_episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
        frameskip: Number of frames to skip between actions
        repeat_action_probability: Probability of repeating previous action
        logger: Optional logger instance
        gif_path: Optional path to save a GIF of the first episode
        gif_fps: Frames per second for the saved GIF (default 30)

    Returns:
        tuple: (mean_reward, std_reward)
    """
    if logger:
        logger.info("Evaluating policy")

    render_mode = "rgb_array" if gif_path else None
    env = env_class(env_name=env_name, render_mode=render_mode,
                    frameskip=frameskip,
                    repeat_action_probability=repeat_action_probability)

    rewards = []
    frames = []

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            record = gif_path and episode == 0

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            step_count = 0
            while True:
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step_count += 1

                if record:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if terminated or truncated:
                    break
                if step_count >= steps_per_episode:
                    if logger:
                        logger.warning(
                            f"Episode {episode} exceeded steps_per_episode limit "
                            f"({steps_per_episode}) without terminating. "
                            f"Consider increasing steps_per_episode."
                        )
                    break

            rewards.append(episode_reward)
    finally:
        env.close()

    if gif_path and frames:
        imageio.mimsave(str(gif_path), frames, fps=gif_fps)
        if logger:
            logger.info(f"Saved evaluation GIF to {gif_path}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

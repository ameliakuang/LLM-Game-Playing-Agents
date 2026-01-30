"""
Shared training utilities for Trace-based agent training scripts.
"""
import numpy as np
import opto.trace as trace


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
                frameskip=1, repeat_action_probability=0.0, logger=None):
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
        
    Returns:
        tuple: (mean_reward, std_reward)
    """
    if logger:
        logger.info("Evaluating policy")
    

    env = env_class(env_name=env_name, render_mode=None,
                    frameskip=frameskip,
                    repeat_action_probability=repeat_action_probability)

    rewards = []
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            for _ in range(steps_per_episode):
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
    finally:
        env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

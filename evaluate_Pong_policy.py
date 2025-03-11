import gymnasium as gym
import ale_py
import numpy as np

from dotenv import load_dotenv
from pong_ocatari_LLM_agent import PongOCAtariTracedEnv, Policy

load_dotenv()
gym.register_envs(ale_py)

def test_policy(env, policy, num_episodes=10, steps_per_episode=4000):
    rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    env = PongOCAtariTracedEnv(render_mode=None,  # No rendering
                               env_name="PongNoFrameskip-v4",
                               obs_mode="obj",
                               hud=False,
                               frameskip=4,
                               repeat_action_probability=0.00)
    policy = Policy()
    policy_ckpt = "trace_ckpt/PongNoFrameskip-v4_20250305_204211_skip4_sticky0.0_horizon400_optimSteps20_mem5/8.pkl"
    policy.load(policy_ckpt)
    for p in policy.parameters():
        print(p.name, p.data)
    try:
        test_policy(env, policy, num_episodes=30)
    finally:
        env.close()

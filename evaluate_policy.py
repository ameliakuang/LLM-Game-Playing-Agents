import gymnasium as gym
import ale_py
import numpy as np

from dotenv import load_dotenv
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from pong_ocatari_LLM_agent import PongOCAtariTracedEnv, Policy as PongPolicy
from breakout_ocatari_LLM_agent import TracedEnv, Policy as BreakoutPolicy

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
    frame_skip = 4
    sticky_action_p = 0.0
    env_name = "BreakoutNoFrameskip-v4"
    horizon = 400

    # load policy
    policy = BreakoutPolicy()
    # 105 on breakout with old env:
    # policy_ckpt = "trace_ckpt/BreakoutNoFrameskip-v4_20250309_001316_skip4_sticky0.0_horizon400_optimSteps30_mem5/28.pkl"
    # 163 on breakout with new env:
    # policy_ckpt = "trace_ckpt/BreakoutNoFrameskip-v4_20250312_021201_skip4_sticky0.0_horizon300_optimSteps100_mem5/9.pkl"
    # policy_ckpt = None
    # 203 on breakout with new env, trained with initial_policy_steps = 1000, initial_policy achieving 163 scores
    # policy_ckpt = "trace_ckpt/BreakoutNoFrameskip-v4_20250312_164217_skip4_sticky0.0_horizon300_optimSteps100_mem5/3.pkl"
    # 329 on breakout with new env, trained with no initial policy stepping
    policy_ckpt = "trace_ckpt/BreakoutNoFrameskip-v4_20250312_190211_skip4_sticky0.0_horizon300_optimSteps100_mem5/1.pkl"

    from trained_policies.Breakout import Policy as InitialPolicy
    # initial_policy = InitialPolicy()
    # initial_policy_steps=1000
    initial_policy = None
    initial_policy_steps=None


    env = TracedEnv(render_mode="human",
                    env_name=env_name,
                    obs_mode="ori",
                    hud=False,
                    frameskip=frame_skip,
                    repeat_action_probability=sticky_action_p,
                    initial_policy=initial_policy,
                    initial_policy_steps=initial_policy_steps)

    if policy_ckpt:
        policy.load(policy_ckpt)
    for p in policy.parameters():
        print(p.name, p.data)
    try:
        test_policy(env, policy, num_episodes=1)
    finally:
        env.close()

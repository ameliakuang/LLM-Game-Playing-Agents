import gymnasium as gym
import ale_py
import numpy as np
import argparse
import os

from dotenv import load_dotenv
from pong_ocatari_LLM_agent import PongOCAtariTracedEnv as PongEnv, Policy as PongPolicy
from breakout_ocatari_LLM_agent import TracedEnv as BreakoutEnv, Policy as BreakoutPolicy
from space_invaders_ocatari_LLM_agent import TracedEnv as SpaceInvadersEnv, Policy as SpaceInvadersPolicy
from best_policies.Pong import Policy as PongBestPolicy
from best_policies.Breakout import Policy as BreakoutBestPolicy
from best_policies.SpaceInvaders import Policy as SpaceInvadersBestPolicy
load_dotenv()
gym.register_envs(ale_py)

def test_policy(env, policy, steps_per_episode=4000):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    for _ in range(steps_per_episode):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        steps += 1
        if terminated or truncated:
            break
 
    # print(f"\nResults over {num_episodes} episodes:")
    print(f"Episode reward: {episode_reward} over {steps} steps")
    return episode_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Policy Given Checkpoints')
    parser.add_argument('--game', type=str, choices=['Pong', 'Breakout', 'SpaceInvaders'], 
                    default='Pong', help='Atari game to evaluate, ["Pong", "Breakout", "SpaceInvaders"]')
    parser.add_argument('--policy_ckpt_dir', type=str, required=True,
                        help='Directory pointing to policy checkpoints')
    parser.add_argument('--ckpt_iter', type=int, required=True,
                        help='Checkpoint iteration number to load')
    parser.add_argument('--render', action='store_true', help='Enable rendering')

    args = parser.parse_args()



    policy_ckpt = os.path.join(args.policy_ckpt_dir, f"{args.ckpt_iter}.pkl")
    if not os.path.exists(policy_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {policy_ckpt}")
    render_mode = None
    if args.render:
        render_mode = "human"

    obs_mode = "obj"
    hud = False
    frameskip = 4
    repeat_action_probability = 0.00
    if args.game == "Pong":
        env = PongEnv(render_mode=render_mode,
                                env_name="PongNoFrameskip-v4",
                                obs_mode=obs_mode,
                                hud=hud,
                                frameskip=frameskip,
                                repeat_action_probability=repeat_action_probability)
        policy = PongPolicy()
        policy.load(policy_ckpt)
    elif args.game == "Breakout":
        env = BreakoutEnv(env_name="BreakoutNoFrameskip-v4",
                          render_mode=render_mode,
                          obs_mode=obs_mode,
                          hud=hud,
                          frameskip=frameskip,
                          repeat_action_probability=repeat_action_probability,
                          initial_policy=None,
                          initial_policy_steps=None)
        policy = BreakoutPolicy()
        policy.load(policy_ckpt)
    elif args.game == "SpaceInvaders":
        env = SpaceInvadersEnv(env_name="SpaceInvadersNoFrameskip-v4",
                               render_mode=render_mode,
                               frameskip=frameskip,
                               repeat_action_probability=repeat_action_probability)
        policy = SpaceInvadersPolicy()
        policy.load(policy_ckpt)
        
    else:
        raise ValueError(f"Invalid game name {args.game}")
    
    for p in policy.parameters():
        print(p.name, p.data)
    try:
        test_policy(env, policy)
    finally:
        env.close()

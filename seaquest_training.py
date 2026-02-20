import os
import ale_py
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import io
import contextlib
import random
import time

from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym
load_dotenv(override=True)

import opto.trace as trace
from opto.trace import bundle, Module
from opto.optimizers.optoprime import OptoPrime
from trace_envs.seaquest import TracedEnv
from logging_util import setup_logger
from training_utils import rollout, evaluate_policy, create_experiment_dir

gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        situation = self.analyze_situation(obs)
        action = self.select_action(situation, obs)
        return action

    @bundle(trainable=True)
    def analyze_situation(self, obs):
        """
        Analyze the submarine's position, nearby enemies, divers, and oxygen
        status to determine the best course of action.

        Game setup:
        - You pilot a submarine in an underwater scene.
        - The screen x range is roughly 5-155, y range roughly 16-180.
        - The water surface is at the top of the screen (low y values).
        - Enemies (killer sharks and enemy subs) swim horizontally across
          the screen at various depths. Colliding with them destroys your sub.
        - Divers float in the water and must be rescued by swimming into them.
        - You can carry up to 6 divers at a time.
        - An oxygen gauge depletes constantly. You MUST surface (move to the
          top of the screen, y near 16) before oxygen runs out.
        - When you surface with 6 divers collected, you receive bonus points.
        - If oxygen runs out, your sub explodes and you lose a life.
        - Surfacing without any divers when oxygen is not critical also loses
          a diver from reserves.

        Scoring:
        - Shooting enemies: 20 pts initially (increases by 10 per surface, max 90)
        - Rescuing divers: 50 pts initially (increases by 50 per surface, max 1000)
        - Oxygen bonus: points for remaining oxygen when surfacing

        Strategy:
        - Rescue divers while shooting enemies that are in the way.
        - Monitor oxygen — surface before it runs out.
        - Collect 6 divers before surfacing for maximum bonus.
        - Shoot enemies from a distance rather than trying to dodge them.

        Args:
            obs (dict): Dictionary with:
                - "Player": {x, y, w, h, dx, dy}
                - "Enemies": list of {x, y, w, h, dx, dy} (may be absent)
                - "Divers": list of {x, y, w, h, dx, dy} (may be absent)
                - "lives": int
                - "reward": float
        Returns:
            dict: Analysis with keys:
                - "player_x": int, sub x position
                - "player_y": int, sub y position
                - "nearest_enemy": dict or None, closest enemy
                - "nearest_diver": dict or None, closest diver to rescue
                - "should_surface": bool, whether oxygen is critically low
        """
        if 'Player' not in obs:
            return {"player_x": 80, "player_y": 100, "nearest_enemy": None,
                    "nearest_diver": None, "should_surface": False}

        player = obs['Player']
        return {"player_x": player['x'], "player_y": player['y'],
                "nearest_enemy": None, "nearest_diver": None,
                "should_surface": False}

    @bundle(trainable=True)
    def select_action(self, situation, obs):
        """
        Select the action to take based on the situation analysis.

        Available actions (18 total — movement + fire combinations):
        - 0:  NOOP
        - 1:  FIRE            (shoot torpedo in facing direction)
        - 2:  UP              (move up / toward surface)
        - 3:  RIGHT           (move right)
        - 4:  LEFT            (move left)
        - 5:  DOWN            (move down / deeper)
        - 6:  UPRIGHT         (move diagonally up-right)
        - 7:  UPLEFT          (move diagonally up-left)
        - 8:  DOWNRIGHT       (move diagonally down-right)
        - 9:  DOWNLEFT        (move diagonally down-left)
        - 10: UPFIRE          (move up + fire)
        - 11: RIGHTFIRE       (move right + fire)
        - 12: LEFTFIRE        (move left + fire)
        - 13: DOWNFIRE        (move down + fire)
        - 14: UPRIGHTFIRE     (move up-right + fire)
        - 15: UPLEFTFIRE      (move up-left + fire)
        - 16: DOWNRIGHTFIRE   (move down-right + fire)
        - 17: DOWNLEFTFIRE    (move down-left + fire)

        Movement Logic:
        - If oxygen is critical (should_surface=True): move UP (2) or UPFIRE
          (10) to reach the surface immediately.
        - If a diver is nearby: move toward the diver to rescue them.
        - If an enemy is in the path: use FIRE combinations (11, 12, or 1)
          to shoot it while moving.
        - Prefer FIRE combinations (10-17) when enemies are nearby to shoot
          while moving — this is more efficient than stopping to fire.
        - When no immediate target, patrol at mid-depth to find divers.
        - Surface (move to top) when you have 6 divers for maximum bonus.

        Args:
            situation (dict): Analysis from analyze_situation.
            obs (dict): Full observation dictionary.
        Returns:
            int: Action index 0-17.
        """
        if situation is None:
            return 1  # Default: fire

        return random.choice([2, 3, 4, 5, 11, 12])


def optimize_policy(
    env_name="SeaquestNoFrameskip-v4",
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    frame_skip=4,
    sticky_action_p=0.00,
    logger=None,
    policy_ckpt=None,
    experiment_dirs=None,
):
    if logger is None:
        logger = setup_logger(__name__, env_name)

    if experiment_dirs is None:
        experiment_dirs = create_experiment_dir("seaquest", timestamp)

    policy = Policy()
    if policy_ckpt:
        logger.info(f"Continuing training from ckpt: {policy_ckpt}")
        policy.load(policy_ckpt)
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size, max_tokens=4096)
    env = TracedEnv(env_name=env_name,
                    frameskip=frame_skip,
                    repeat_action_probability=sticky_action_p)
    perf_csv_filename = experiment_dirs["perf_csv"]
    trace_ckpt_dir = experiment_dirs["trace_ckpt_dir"]
    gif_dir = experiment_dirs["gif_dir"]
    try:
        rewards = []
        optimization_data = []
        logger.info("Optimization Starts")
        best_mean_reward = 0
        best_ckpt = None
        best_iter = None
        recent_mean_rewards = []
        for i in range(n_optimization_steps):
            mean_rewards = np.nan
            std_rewards = np.nan
            steps_used = np.nan
            step_start_time = time.time()
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                num_episodes = 1
                steps_per_episode = 4000
                gif_path = gif_dir / f"eval_iter_{i}.gif"
                mean_rewards, std_rewards = evaluate_policy(policy,
                                                        TracedEnv,
                                                        env_name,
                                                        num_episodes=num_episodes,
                                                        steps_per_episode=steps_per_episode,
                                                        frameskip=frame_skip,
                                                        repeat_action_probability=sticky_action_p,
                                                        logger=logger,
                                                        gif_path=gif_path)
                steps_used = traj['steps']

                recent_mean_rewards.append(mean_rewards)
                if len(recent_mean_rewards) > 5:
                    recent_mean_rewards.pop(0)
                if mean_rewards >= 5000:
                    logger.info(f"Congratulations! You've achieved a score of {mean_rewards} with std dev {std_rewards}. Ending optimization early.")
                    rewards.append(sum(traj['rewards']))
                    optimization_data.append({
                        "Optimization Step": i,
                        "Mean Reward": mean_rewards,
                        "Std Dev Reward": std_rewards,
                        "Wall Clock Time (s)": time.time() - step_start_time,
                        "Training Steps": steps_used,
                        "Max Training Steps": horizon,
                    })
                    df = pd.DataFrame(optimization_data)
                    df.to_csv(perf_csv_filename, index=False)
                    policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))
                    break
                if mean_rewards >= 2000:
                    feedback += (f"\nGreat progress! You're scoring {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Keep rescuing divers and shooting enemies. Surface with 6 divers for maximum bonus.")
                elif mean_rewards > 0:
                    feedback += (f"\nYou're scoring {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Rescue divers by swimming into them (50+ pts each). "
                                 f"Shoot enemies with FIRE combinations to clear the path (20+ pts each). "
                                 f"Surface (move to top) before oxygen runs out — running out of oxygen kills you. "
                                 f"Collect 6 divers before surfacing for maximum bonus points.")
                elif mean_rewards <= 0:
                    feedback += (f"\nYour score is {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"You must rescue divers and shoot enemies to score. "
                                 f"CRITICAL: surface before oxygen depletes or you lose a life. "
                                 f"Avoid colliding with enemies — shoot them from a distance.")
                target = traj['observations'][-1]

                rewards.append(sum(traj['rewards']))

                # only save ckpt of policies without syntax/running error
                policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))
                # Update the best checkpoint if the current mean reward is higher
                if mean_rewards > best_mean_reward:
                    best_mean_reward = mean_rewards
                    best_ckpt = os.path.join(trace_ckpt_dir, f"{i}.pkl")
                    best_iter = i
                    logger.info(f"New best checkpoint saved at {best_ckpt}")
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node


            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}")

            instruction = "In Seaquest, you pilot a submarine rescuing divers and fighting enemies underwater. "
            instruction += "Actions: 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN, "
            instruction += "6=UPRIGHT, 7=UPLEFT, 8=DOWNRIGHT, 9=DOWNLEFT, "
            instruction += "10=UPFIRE, 11=RIGHTFIRE, 12=LEFTFIRE, 13=DOWNFIRE, "
            instruction += "14=UPRIGHTFIRE, 15=UPLEFTFIRE, 16=DOWNRIGHTFIRE, 17=DOWNLEFTFIRE. "
            instruction += "Rescue divers by swimming into them (50+ pts each, increases per surface). "
            instruction += "Shoot enemies — sharks and enemy subs (20+ pts each, increases per surface). "
            instruction += "CRITICAL: oxygen depletes constantly. You MUST surface (move to top of screen) before it runs out or you lose a life. "
            instruction += "Collect 6 divers then surface for maximum bonus points (gold ingots). "
            instruction += "Colliding with enemies destroys your sub — shoot them from a distance. "
            instruction += "You start with 4 submarines. Extra sub every 10,000 pts (max 6 reserves). "
            instruction += "Use FIRE combinations (actions 10-17) to shoot while moving — more efficient than stopping to fire. "
            instruction += "Strategy: patrol mid-depth to find divers, shoot approaching enemies, surface when oxygen is low or when 6 divers collected. "
            instruction += "Analyze the trace to understand when you lose lives (oxygen depletion or enemy collision) and optimize to rescue more divers."
            optimizer.objective = optimizer.default_objective + instruction

            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=verbose)
                llm_output = stdout_buffer.getvalue()
                if llm_output:
                    logger.info(f"LLM response:\n {llm_output}")

            logger.info(f"Iteration: {i}, Feedback: {feedback}")
            optimization_data.append({
                    "Optimization Step": i,
                    "Mean Reward": mean_rewards,
                    "Std Dev Reward": std_rewards,
                    "Wall Clock Time (s)": time.time() - step_start_time,
                    "Training Steps": steps_used,
                    "Max Training Steps": horizon,
                })
            df = pd.DataFrame(optimization_data)
            df.to_csv(perf_csv_filename, index=False)

            if error:
                # Load the latest policy checkpoint from the trace_ckpt_dir
                latest_checkpoint = max([int(f.split('.')[0]) for f in os.listdir(trace_ckpt_dir) if f.endswith('.pkl')])
                ckpt_path = os.path.join(trace_ckpt_dir, f"{latest_checkpoint}.pkl")
                logger.info(f"Loading ckpt of {ckpt_path}")
                policy.load(ckpt_path)

            # Check if the performance has dropped significantly in the recent 5 iterations
            if best_iter and i > best_iter + 5 and recent_mean_rewards[-1] < 0.8 * best_mean_reward:
                logger.info("Performance has dropped significantly in the recent 5 iterations. Loading the best checkpoint so far.")
                policy.load(best_ckpt)
    finally:
        if env is not None:
            env.close()

    logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seaquest AI training")
    parser.add_argument("--env-name", type=str, default="SeaquestNoFrameskip-v4")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--n-optimization-steps", type=int, default=30)
    parser.add_argument("--memory-size", type=int, default=5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--sticky-action-p", type=float, default=0.0)
    parser.add_argument("--policy-ckpt", type=str, default=None, help="Path to policy checkpoint to resume from")
    args = parser.parse_args()

    frame_skip = args.frame_skip
    sticky_action_p = args.sticky_action_p
    env_name = args.env_name
    horizon = args.horizon
    n_optimization_steps = args.n_optimization_steps
    memory_size = args.memory_size
    policy_ckpt = args.policy_ckpt

    # Create per-experiment directory
    experiment_dirs = create_experiment_dir(f"seaquest_horizon{horizon}", timestamp)

    # Set up logging
    logger = setup_logger(
        __name__,
        env_name,
        timestamp=timestamp,
        frame_skip=frame_skip,
        sticky_action_p=sticky_action_p,
        horizon=horizon,
        optim_steps=n_optimization_steps,
        memory_size=memory_size,
        log_file=experiment_dirs["log_file"],
        prefix="OCAtari"
    )

    logger.info("Starting Seaquest AI training...")
    rewards = optimize_policy(
        env_name=env_name,
        horizon=horizon,
        n_optimization_steps=n_optimization_steps,
        memory_size=memory_size,
        verbose='output',
        frame_skip=frame_skip,
        sticky_action_p=sticky_action_p,
        logger=logger,
        policy_ckpt=policy_ckpt,
        experiment_dirs=experiment_dirs,
    )
    logger.info("Training completed.")

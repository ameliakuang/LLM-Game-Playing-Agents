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
from trace_envs.asterix import TracedEnv
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
        Analyze the game state to find the best item to collect and detect
        nearby dangers.

        Game layout:
        - 8 horizontal lanes at y-positions 26, 42, 58, 74, 90, 106, 122, 138
          (16px apart).
        - Objects move horizontally across lanes. Each lane has at most one object.
        - Player can move in 8 directions + NOOP. Screen x range: 0-152.

        Scoring:
        - Collecting a Consumable gives 50+ points; Reward gives bonus points.
        - Touching an Enemy (lyre) costs a life. Losing all lives ends the game.
        - An enemy in the same lane is only dangerous if the player is near its
          x-position — being in the same lane but far away horizontally is safe.

        Observation fields:
        - "Player": {x, y} — player position.
        - "Enemies": list of {x, y, type}, sorted by distance. May be absent.
        - "Consumables": list of {x, y, type}, sorted by distance. May be absent.
        - "Rewards": list of {x, y, type}, sorted by distance. May be absent.
        - "lives": int, remaining lives.
        - "reward": float, reward from the previous step.

        Strategy hints:
        - Use proximity-based danger detection: an enemy is dangerous only if
          it's within ~30px horizontally. Track which lane y-positions are
          dangerous and which enemies are approaching (<20px).
        - When selecting which item to pursue, factor in: Manhattan distance,
          item type (Rewards are worth more), whether the item's lane is
          dangerous, and whether the item is in the player's current lane
          (cheaper to reach).

        Args:
            obs (dict): The observation dictionary.
        Returns:
            dict with keys:
                - "nearest_item": dict {x, y, type} of item to pursue, or None
                - "player_x": int
                - "player_y": int
                - "dangerous_lanes": list of y-positions with nearby enemies
                - "approaching_enemies": list of very close enemies, or None
        """
        if 'Player' not in obs:
            return {"nearest_item": None, "player_x": 76, "player_y": 74,
                    "dangerous_lanes": [], "approaching_enemies": None}

        player = obs['Player']
        items = obs.get('Consumables', []) + obs.get('Rewards', [])
        nearest_item = items[0] if items else None

        return {
            "nearest_item": nearest_item,
            "player_x": player['x'],
            "player_y": player['y'],
            "dangerous_lanes": [],
            "approaching_enemies": None,
        }

    @bundle(trainable=True)
    def select_action(self, situation, obs):
        """
        Select an action to move the player toward the nearest item while
        avoiding enemies.

        Available actions (action index -> effect):
        - 0: NOOP      — stay in place
        - 1: UP         — move toward smaller y
        - 2: RIGHT      — move toward larger x
        - 3: LEFT       — move toward smaller x
        - 4: DOWN       — move toward larger y
        - 5: UPRIGHT    — move up and right simultaneously
        - 6: UPLEFT     — move up and left simultaneously
        - 7: DOWNRIGHT  — move down and right simultaneously
        - 8: DOWNLEFT   — move down and left simultaneously

        Lane y-positions: 26, 42, 58, 74, 90, 106, 122, 138 (16px apart).
        Diagonal actions (5-8) move in two directions at once and are more
        efficient when the target is in a different lane.

        Strategy hints:
        - If an enemy is approaching in the player's lane, evade to an
          adjacent safe lane (use diagonals to also move toward the target).
        - When in the same lane as the target, move horizontally toward it.
        - When in a different lane, prefer diagonal movement to close both
          the x and y gap simultaneously.
        - Be willing to enter a slightly dangerous lane if a high-value
          Reward is very close.

        Args:
            situation (dict): Analysis from analyze_situation with keys
                nearest_item, player_x, player_y, dangerous_lanes,
                approaching_enemies.
            obs (dict): Full observation dictionary.
        Returns:
            int: Action index 0-8.
        """
        if situation is None or situation.get('nearest_item') is None:
            return 0

        return random.choice([1, 2, 3, 4, 5, 6, 7, 8])


def optimize_policy(
    env_name="AsterixNoFrameskip-v4",
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
        experiment_dirs = create_experiment_dir("asterix", timestamp)

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
                num_episodes = 3
                steps_per_episode = 20000
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
                    feedback += (f"\nGood: scoring {mean_rewards:.0f} pts avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Consider whether you can collect items more efficiently or take advantage of lanes you might be avoiding unnecessarily.")
                elif mean_rewards > 0:
                    feedback += (f"\nScoring {mean_rewards:.0f} pts avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Points come from collecting Consumables and Rewards. Touching an Enemy costs a life. "
                                 f"Items in the observation lists are sorted by distance (nearest first).")
                elif mean_rewards <= 0:
                    feedback += (f"\nScoring {mean_rewards:.0f} pts avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Points are only earned by collecting Consumable and Reward items. "
                                 f"Move toward items to collect them — compare player x/y with item x/y to pick the right direction.")
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

            instruction = "In Asterix, you control a player across 8 horizontal lanes (lane 0-7) to collect items and avoid enemies. "
            instruction += "Actions: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN, 5=UPRIGHT, 6=UPLEFT, 7=DOWNRIGHT, 8=DOWNLEFT. "
            instruction += "Lane y-positions (top to bottom): 26, 42, 58, 74, 90, 106, 122, 138 (16px apart). Screen x: 0-152. "
            instruction += "Each lane has at most one object: Consumable (collect for 50-500pts), Reward (collect for bonus pts), or Enemy/lyre (touching costs a life). "
            instruction += "The observation provides 'lane' (0-7) and 'dist' (Manhattan distance) for each object, "
            instruction += "'safe_collectible_lanes' (lanes with items but no enemy), and 'enemy_lanes'. "
            instruction += "Object lists are sorted by distance (nearest first). "
            instruction += "Score by collecting items. Enemies only matter if you're close to their x-position in the same lane. "
            instruction += "Diagonal actions (5-8) move in two directions at once."
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
    parser = argparse.ArgumentParser(description="Asterix AI training")
    parser.add_argument("--env-name", type=str, default="AsterixNoFrameskip-v4")
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
    experiment_dirs = create_experiment_dir(f"asterix_horizon{horizon}", timestamp)

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

    logger.info("Starting Asterix AI training...")
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

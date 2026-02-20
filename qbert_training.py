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
from trace_envs.qbert import TracedEnv
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
        Analyze Q*bert's position on the pyramid and the positions of enemies
        to determine safe movement directions.

        Game setup:
        - Q*bert hops on a pyramid of cubes viewed isometrically.
        - The pyramid has 7 rows. The top cube is row 1 (1 cube), bottom row
          is row 7 (7 cubes).
        - Movement is diagonal: UP=upper-right, LEFT=upper-left,
          RIGHT=lower-right, DOWN=lower-left on the pyramid.
        - Hopping on a cube changes its color. The goal is to change ALL cubes
          to the destination color.
        - Jumping off the pyramid edge kills Q*bert (costs a life), UNLESS
          landing on a flying disk which transports Q*bert back to the top.
        - The screen x range is roughly 16-144, y range roughly 30-190.
        - Q*bert starts at the top of the pyramid.

        Enemies:
        - Red/Purple Balls: bounce down the pyramid, avoid their path.
        - Coily (snake): chases Q*bert, can be lured off the edge via a
          flying disk for 500 bonus points.
        - Sam: hops around reverting cube colors — catch him for 300 points.
        - Green Ball: freezes all enemies temporarily — touch for 100 points.

        Strategy:
        - Identify which enemies are near Q*bert's position.
        - Determine safe directions (no enemies, no edge) to hop toward.
        - Prioritize changing unvisited cubes to the destination color.
        - Use flying disks to escape when Coily is close.

        Args:
            obs (dict): Dictionary with:
                - "Player": {x, y, w, h, dx, dy}
                - "Enemies": list of {x, y, w, h, dx, dy} (may be absent)
                - "Friendlies": list of {x, y, w, h, dx, dy} (may be absent)
                - "lives": int
                - "reward": float
        Returns:
            dict: Analysis with keys:
                - "player_x": int, Q*bert x position
                - "player_y": int, Q*bert y position
                - "nearest_enemy": dict or None
                - "nearest_friendly": dict or None
                - "danger_above": bool, enemy approaching from above
        """
        if 'Player' not in obs:
            return {"player_x": 80, "player_y": 50, "nearest_enemy": None,
                    "nearest_friendly": None, "danger_above": False}

        player = obs['Player']
        return {"player_x": player['x'], "player_y": player['y'],
                "nearest_enemy": None, "nearest_friendly": None,
                "danger_above": False}

    @bundle(trainable=True)
    def select_action(self, situation, obs):
        """
        Select the action to take based on the situation analysis.

        Available actions (movement is diagonal on the isometric pyramid):
        - 0: NOOP         (do nothing)
        - 1: FIRE          (jump onto flying disk if adjacent to one)
        - 2: UP            (hop to upper-right cube on pyramid)
        - 3: RIGHT         (hop to lower-right cube on pyramid)
        - 4: LEFT          (hop to upper-left cube on pyramid)
        - 5: DOWN          (hop to lower-left cube on pyramid)

        Movement Logic:
        - Move toward cubes that haven't been changed to the destination color.
        - Prefer directions away from enemies (Coily, red/purple balls).
        - NEVER hop off the pyramid edge unless a flying disk is there.
        - If Coily is chasing and a flying disk is available, use FIRE or
          hop toward the disk edge to lure Coily off for 500 bonus points.
        - Catch Sam (300 pts) and Green Balls (100 pts) when safe to do so.

        Edge awareness:
        - At the top (row 1): only RIGHT (lower-right) and DOWN (lower-left)
          are safe.
        - On the left edge: UP and DOWN lead off the pyramid.
        - On the right edge: UP and RIGHT lead off the pyramid.

        Args:
            situation (dict): Analysis from analyze_situation.
            obs (dict): Full observation dictionary.
        Returns:
            int: Action index 0-5.
        """
        if situation is None:
            return 0

        return random.choice([2, 3, 4, 5])


def optimize_policy(
    env_name="QbertNoFrameskip-v4",
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
        experiment_dirs = create_experiment_dir("qbert", timestamp)

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
                if mean_rewards >= 10000:
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
                if mean_rewards >= 5000:
                    feedback += (f"\nGreat progress! You're scoring {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Keep changing cubes to the destination color while avoiding enemies.")
                elif mean_rewards > 0:
                    feedback += (f"\nYou're scoring {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Hop on cubes to change them to the destination color (25 pts each). "
                                 f"Avoid enemies — Coily chases you, red/purple balls bounce down. "
                                 f"Catch Sam (300 pts) and green balls (100 pts) when safe. "
                                 f"Do NOT jump off the pyramid edge — it costs a life.")
                elif mean_rewards <= 0:
                    feedback += (f"\nYour score is {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"You must hop on cubes to change their color and score points. "
                                 f"Avoid jumping off the pyramid edges and avoid enemies.")
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

            instruction = "In Q*bert, you control Q*bert hopping on an isometric pyramid of cubes. "
            instruction += "Actions: 0=NOOP, 1=FIRE (use flying disk), 2=UP (upper-right), 3=RIGHT (lower-right), 4=LEFT (upper-left), 5=DOWN (lower-left). "
            instruction += "All movement is diagonal on the pyramid. "
            instruction += "The goal is to change every cube to the destination color by hopping on it (25 pts per cube). "
            instruction += "Completing a round (all cubes correct color) earns a 3100-point bonus. "
            instruction += "Enemies: Red/Purple Balls bounce down the pyramid — avoid them. "
            instruction += "Coily (snake) chases Q*bert — lure him off the edge via flying disks for 500 pts. "
            instruction += "Sam reverts cube colors — catch him for 300 pts. Green Ball freezes enemies — touch for 100 pts. "
            instruction += "CRITICAL: jumping off the pyramid edge without a flying disk kills Q*bert. "
            instruction += "You start with 4 lives. Bonus lives at certain score thresholds. "
            instruction += "Edge awareness: at the top only RIGHT and DOWN are safe; on left edge avoid UP/DOWN; on right edge avoid UP/RIGHT. "
            instruction += "Strategy: systematically hop on cubes to change colors, avoid enemies, use flying disks to escape Coily. "
            instruction += "Analyze the trace to understand when Q*bert dies and optimize movement to color more cubes safely."
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
    parser = argparse.ArgumentParser(description="Qbert AI training")
    parser.add_argument("--env-name", type=str, default="QbertNoFrameskip-v4")
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
    experiment_dirs = create_experiment_dir(f"qbert_horizon{horizon}", timestamp)

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

    logger.info("Starting Qbert AI training...")
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

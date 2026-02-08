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
from trace_envs.freeway import TracedEnv
from logging_util import setup_logger
from training_utils import rollout, evaluate_policy, create_experiment_dir

gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        traffic_analysis = self.analyze_traffic(obs)
        action = self.select_action(traffic_analysis, obs)
        return action

    @bundle(trainable=True)
    def analyze_traffic(self, obs):
        """
        Analyze car positions and velocities to determine which lanes are safe
        to cross right now, based on the chicken's current y-position.

        Game setup:
        - The chicken starts at the bottom (y~187) and must reach the top (y~0).
        - Moving UP decreases y; moving DOWN increases y.
        - There are 10 lanes of traffic at fixed y-positions (top to bottom):
            Lane 1: y=27,  Lane 2: y=43,  Lane 3: y=59,  Lane 4: y=75,  Lane 5: y=91,
            Lane 6: y=107, Lane 7: y=123, Lane 8: y=139, Lane 9: y=155, Lane 10: y=171
        - Top lanes (1-5) have cars moving right-to-left (negative dx).
        - Bottom lanes (6-10) have cars moving left-to-right (positive dx).
        - Each car has width w=8. The chicken has width w=6.
        - Screen x range: roughly 0 to 160.

        Strategy:
        - Identify which lane the chicken is about to enter (the next lane above).
        - Check if there is a car in that lane whose x position would collide with
          the chicken (x~44) within the next few frames.
        - A lane is "safe" if the car's x is far enough from the chicken's x that
          the chicken can pass through before the car arrives.

        Args:
            obs (dict): Dictionary with "Chicken" state {x, y, w, h, dx, dy}
                       and "Cars" list of car states [{x, y, w, h, dx, dy}, ...].
        Returns:
            dict: Analysis with keys:
                - "safe": bool, whether it is safe to move up right now
                - "next_lane_car": dict or None, the car in the next lane above
                - "chicken_y": int, current chicken y position
        """
        if 'Chicken' not in obs:
            return {"safe": True, "next_lane_car": None, "chicken_y": 187}

        chicken_y = obs['Chicken']['y']
        return {"safe": True, "next_lane_car": None, "chicken_y": chicken_y}

    @bundle(trainable=True)
    def select_action(self, traffic_analysis, obs):
        """
        Select the action to take based on the traffic analysis.

        Movement Logic:
        - If the next lane is safe: Move UP (action 1) to make progress
        - If the next lane is NOT safe: NOOP (action 0) to wait for a gap
        - Only use DOWN (action 2) as a last resort to dodge an imminent collision

        The goal is to maximize the number of complete crossings within the time limit.
        Prioritize forward progress: move UP whenever there is a safe gap.

        Args:
            traffic_analysis (dict): Analysis from analyze_traffic with keys:
                - "safe": bool, whether it is safe to move up
                - "next_lane_car": dict or None, the car in the next lane
                - "chicken_y": int, current chicken y position
            obs (dict): Full observation dictionary.
        Returns:
            int: 0 for NOOP, 1 for UP, 2 for DOWN
        """
        if traffic_analysis is None:
            return 1  # Default: move up

        if traffic_analysis.get("safe", True):
            return 1  # UP - safe to advance
        else:
            return 0  # NOOP - wait for gap


def optimize_policy(
    env_name="FreewayNoFrameskip-v4",
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
        experiment_dirs = create_experiment_dir("freeway", timestamp)

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
                steps_per_episode = 2500
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
                if mean_rewards >= 50:
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
                if mean_rewards >= 25:
                    feedback += (f"\nGreat progress! You're scoring {mean_rewards} crossings on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Keep optimizing gap detection to push higher.")
                elif mean_rewards > 0:
                    feedback += (f"\nYou're scoring {mean_rewards} crossings on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Try to improve gap detection and timing to cross more lanes safely. "
                                 f"Remember: always be moving UP unless a car is directly in the way.")
                elif mean_rewards <= 0:
                    feedback += (f"\nYour score is {mean_rewards} crossings on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"The chicken needs to move UP (action 1) to cross lanes. "
                                 f"Wait with NOOP (action 0) only when a car is about to hit you, otherwise keep moving UP.")
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

            instruction = "In Freeway, you guide a chicken from the bottom of the screen (y~187) to the top (y~0), crossing 10 lanes of traffic. "
            instruction += "You score 1 point each time the chicken reaches the top and it resets to the bottom to cross again. "
            instruction += "Actions: 0=NOOP (wait), 1=UP (move toward goal), 2=DOWN (move away from goal). "
            instruction += "There are 10 lanes of cars at fixed y-positions (top to bottom): y=27, 43, 59, 75, 91, 107, 123, 139, 155, 171. "
            instruction += "Top lanes (y=27-91) have cars moving right-to-left (negative dx). Bottom lanes (y=107-171) have cars moving left-to-right (positive dx). "
            instruction += "The chicken is at x~44. Cars have width 8, the chicken has width 6. "
            instruction += "If hit by a car, the chicken is pushed back one lane (novice difficulty). "
            instruction += "The game runs on a ~2 minute timer. Maximize the number of crossings within the time limit. "
            instruction += "Key strategy: always be moving UP unless a car is about to collide. Waiting too long wastes time. "
            instruction += "Analyze the trace to figure out when the chicken gets hit and optimize gap detection to avoid cars while maintaining forward progress."
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
    finally:
        if env is not None:
            env.close()

    logger.info(f"Final Average Reward during Training (Not evaluation so reference only): {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    frame_skip = 4
    sticky_action_p = 0.0
    env_name = "FreewayNoFrameskip-v4"
    horizon = 150
    n_optimization_steps = 30
    memory_size = 1
    policy_ckpt = None

    # Create per-experiment directory
    experiment_dirs = create_experiment_dir("freeway", timestamp)

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

    logger.info("Starting Freeway AI training...")
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

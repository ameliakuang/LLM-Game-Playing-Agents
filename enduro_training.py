import os
import ale_py
import datetime
import numpy as np
import pandas as pd
import io
import contextlib
import random
import time

from dotenv import load_dotenv
import gymnasium as gym
load_dotenv(override=True)

import opto.trace as trace
from opto.trace import bundle, Module
from opto.optimizers.optoprime import OptoPrime
from trace_envs.enduro import TracedEnv
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
        Analyze the positions of enemy cars relative to the player and decide
        which direction to steer (if any) to avoid collisions.

        Coordinate system:
        - The screen is ~160px wide. The player's car is near x=77, y=135-145.
        - Enemy cars approach from ahead (smaller y) toward the player (larger y).
        - Both player and enemy cars are ~16px wide.
        - The road curves, shifting all car x-positions over time.

        Scoring:
        - +1 for each car you overtake (pass), -1 for each car that overtakes you.
        - Collisions slow you down dramatically, causing nearby cars to pass you.
        - Avoiding collisions is critical to maintaining speed and scoring.

        Observation fields:
        - "Player": {x, y, w, h} — player car position and size.
        - "EnemyCars": list of enemy cars sorted by proximity (largest y first).
            Each has {x, y, rel_x, rel_y, w, h}.
            rel_x = enemy_x - player_x (negative means enemy is to the left).
            rel_y = enemy_y - player_y (negative means enemy is ahead/above,
            positive means enemy is close or behind/below).
          May be absent if no enemy cars are visible.
        - "nearest_enemy_rel_x": rel_x of the closest enemy car.
        - "nearest_enemy_rel_y": rel_y of the closest enemy car.
        - "reward": float, reward from the previous step.

        Steering principle: steer away from an enemy that is laterally close.
        If an enemy's rel_x > 0 (enemy is to the right), steer left.
        If an enemy's rel_x < 0 (enemy is to the left), steer right.

        Be careful not to over-steer: only react to enemies that are actually
        close enough to collide. Reacting to distant cars causes unnecessary
        swerving that can lead to worse collisions.

        Args:
            obs (dict): The observation dictionary.
        Returns:
            dict with keys:
                - "steer_direction": "left", "right", or "none"
                - "threat_close": bool
                - "player_x": int
                - "player_y": int
        """
        if 'Player' not in obs:
            return {"steer_direction": "none", "threat_close": False,
                    "player_x": 77, "player_y": 140}

        player = obs['Player']
        steer = "none"
        threat = False

        if 'nearest_enemy_rel_x' in obs and 'nearest_enemy_rel_y' in obs:
            rel_x = obs['nearest_enemy_rel_x']
            rel_y = obs['nearest_enemy_rel_y']
            # React to the nearest enemy if it's close vertically and laterally
            if abs(rel_y) < 25 and abs(rel_x) < 18:
                threat = True
                steer = "left" if rel_x > 0 else "right"

        return {
            "steer_direction": steer,
            "threat_close": threat,
            "player_x": player['x'],
            "player_y": player['y'],
        }

    @bundle(trainable=True)
    def select_action(self, situation, obs):
        """
        Select the best action based on the situation analysis.

        Available actions (action index → effect):
        - 0: NOOP          — coast, no acceleration (loses speed over time)
        - 1: FIRE           — accelerate forward (maintains/increases speed)
        - 2: RIGHT          — steer right without accelerating
        - 3: LEFT           — steer left without accelerating
        - 4: DOWN           — brake (slows down significantly)
        - 5: DOWNRIGHT      — brake + steer right
        - 6: DOWNLEFT       — brake + steer left
        - 7: RIGHTFIRE      — steer right + accelerate (dodge right while keeping speed)
        - 8: LEFTFIRE       — steer left + accelerate (dodge left while keeping speed)

        Key considerations:
        - Speed is essential for overtaking cars. Actions that include FIRE
          (1, 7, 8) maintain speed while acting.
        - Collisions slow the player drastically and cause nearby cars to
          overtake you, each costing -1 reward.
        - Braking (4, 5, 6) and coasting (0) reduce speed, making it harder
          to pass cars. Prefer steering+accelerating (7, 8) over pure steering.

        Args:
            situation (dict): Analysis from analyze_situation.
            obs (dict): Full observation dictionary.
        Returns:
            int: Action index 0-8.
        """
        if situation is None:
            return 1  # accelerate

        if situation.get('threat_close'):
            if situation['steer_direction'] == 'left':
                return 8  # LEFTFIRE — dodge left while accelerating
            elif situation['steer_direction'] == 'right':
                return 7  # RIGHTFIRE — dodge right while accelerating

        return 1  # FIRE — accelerate when path is clear


def optimize_policy(
    env_name="EnduroNoFrameskip-v4",
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
        experiment_dirs = create_experiment_dir("enduro", timestamp)

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
                feedback = f"Episode ends after {traj['steps']} steps with net cars passed: {sum(traj['rewards']):.0f}"
                num_episodes = 1
                steps_per_episode = 5000
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
                if mean_rewards >= 200:
                    logger.info(f"Congratulations! Net cars passed: {mean_rewards:.0f} (day 1 quota met) with std dev {std_rewards:.0f}. Ending optimization early.")
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
                if mean_rewards >= 100:
                    feedback += (f"\nGreat progress! Net cars passed: {mean_rewards:.0f} avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Keep accelerating and dodging to avoid collisions (negative reward) and pass more cars to reach the 200-car quota for day 1.")
                elif mean_rewards > 0:
                    feedback += (f"\nNet cars passed: {mean_rewards:.0f} avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Score = cars you overtake minus cars that overtake you. "
                                 f"Collisions cause negative reward because they slow you down and cars pass you. "
                                 f"Maintain top speed with FIRE (1) and dodge with RIGHTFIRE (7) / LEFTFIRE (8).")
                elif mean_rewards <= 0:
                    feedback += (f"\nNet cars passed: {mean_rewards:.0f} avg over {num_episodes} episodes (std={std_rewards:.0f}). "
                                 f"Negative or zero means more cars are overtaking you than you are passing. "
                                 f"You MUST accelerate (FIRE, action 1) and avoid collisions — each collision slows you "
                                 f"and causes cars to pass you (-1 reward each). Steer with RIGHTFIRE (7) / LEFTFIRE (8).")
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

            instruction = "In Enduro, you race a car and score by overtaking enemy cars (+1 each) while avoiding being overtaken (-1 each). "
            instruction += "Collisions slow the player drastically, causing nearby cars to pass you (multiple -1 penalties). "
            instruction += "Actions: 0=NOOP, 1=FIRE(accelerate), 2=RIGHT, 3=LEFT, 4=DOWN(brake), "
            instruction += "5=DOWNRIGHT, 6=DOWNLEFT, 7=RIGHTFIRE(steer right+accelerate), 8=LEFTFIRE(steer left+accelerate). "
            instruction += "The observation has rel_x (enemy_x - player_x; negative=enemy left, positive=enemy right) "
            instruction += "and rel_y (enemy_y - player_y; negative=enemy ahead with smaller y, positive=enemy close/behind with larger y). "
            instruction += "EnemyCars list is sorted by proximity (largest y first). "
            instruction += "Maintain speed (prefer actions with FIRE: 1, 7, 8) and steer away from close enemies to avoid collisions. "
            instruction += "Be precise about which enemies are actual collision threats — over-steering wastes speed and causes worse outcomes."
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
    parser = argparse.ArgumentParser(description="Enduro AI training")
    parser.add_argument("--env-name", type=str, default="EnduroNoFrameskip-v4")
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
    experiment_dirs = create_experiment_dir(f"enduro_horizon{horizon}", timestamp)

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

    logger.info("Starting Enduro AI training...")
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

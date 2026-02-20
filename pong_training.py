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
load_dotenv(override=True)
from autogen import config_list_from_json
import gymnasium as gym 
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from trace_envs.pong import PongOCAtariTracedEnv
from logging_util import setup_logger
from training_utils import rollout, evaluate_policy, create_experiment_dir


gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        predicted_ball_y = self.predict_ball_trajectory(obs)
        action = self.select_action(predicted_ball_y, obs)
        return action

    @bundle(trainable=True)
    def predict_ball_trajectory(self, obs):
        """
        Predict the y-coordinate where the ball will intersect with the player's paddle by calculating its trajectory,
        using ball's (x, y) and (dx, dy) and accounting for bounces off the top and bottom walls.

        Game Setup:
        - Screen dimensions: The game screen has boundaries where the ball bounces
          - Top boundary: y=30
          - Bottom boundary: y=190
        - Paddle positions:
          - Player paddle: right side of screen (x = 140)
          - Enemy paddle: left side of screen (x = 16)

        Args:
            obs (dict): Dictionary containing object states for "Player", "Ball", and "Enemy".
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).

        Returns:
            float: Predicted y-coordinate where the ball will intersect the player's paddle plane.
                  Returns None if ball position cannot be determined.

        """
        if 'Ball' in obs:
            return obs['Ball'].get("y", None)
        return None
    
    @bundle(trainable=True)
    def select_action(self, predicted_ball_y, obs):
        '''
        Select the optimal action to move player paddle by comparing current player position and predicted_ball_y.
        
        IMPORTANT! Movement Logic:
        - If the player paddle's y position is GREATER than predicted_ball_y: Move DOWN (action 2)
          (because the paddle needs to move downward to meet the ball)
        - If the player paddle's y position is LESS than predicted_ball_y: Move UP (action 3)
          (because the paddle needs to move upward to meet the ball)
        - If the player paddle is already aligned with predicted_ball_y: NOOP (action 0)
          (to stabilize the paddle when it's in position)
        Ensure stable movement to avoid missing the ball when close by.

        Args:
            predicted_ball_y (float): predicted y coordinate of the ball or None
            obs(dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
        Returns:
            int: 0 for NOOP, 2 for DOWN, 3 for UP
        '''

        if predicted_ball_y is not None and 'Player' in obs:
            return random.choice([2, 3])
        return 0
            



def optimize_policy(
    env_name="PongNoFrameskip-v4",
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    frame_skip=4,
    sticky_action_p=0.00,
    logger=None,
    experiment_dirs=None,
):
    if logger is None:
        logger = setup_logger(__name__, env_name)

    if experiment_dirs is None:
        experiment_dirs = create_experiment_dir("pong", timestamp)

    policy = Policy()
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    env = PongOCAtariTracedEnv(env_name=env_name,
                               frameskip=frame_skip,
                               repeat_action_probability=sticky_action_p)
    perf_csv_filename = experiment_dirs["perf_csv"]
    trace_ckpt_dir = experiment_dirs["trace_ckpt_dir"]
    gif_dir = experiment_dirs["gif_dir"]
    try:
        rewards = []
        optimization_data = []
        logger.info("Optimization Starts")
        for i in range(n_optimization_steps):
            step_start_time = time.time()
            mean_rewards = np.nan
            std_rewards = np.nan
            steps_used = np.nan
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                gif_path = gif_dir / f"eval_iter_{i}.gif"
                mean_rewards, std_rewards = evaluate_policy(policy,
                                                        PongOCAtariTracedEnv,
                                                        env_name,
                                                        num_episodes=10,
                                                        frameskip=frame_skip,
                                                        repeat_action_probability=sticky_action_p,
                                                        logger=logger,
                                                        gif_path=gif_path)
                steps_used = traj['steps']
                if mean_rewards >= 21:
                    logger.info(f"Congratulations! You've achieved a perfect score of {mean_rewards} with std dev {std_rewards}. Ending optimization early.")
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
                if mean_rewards >= 19:
                    feedback += (f"\nGood job! You're close to winning the game! "
                                 f"You're scoring {mean_rewards} points against the opponent on average of 10 games with std dev {std_rewards}, "
                                 f"only {21-mean_rewards} points short of winning.")
                elif mean_rewards > 0:
                    feedback += (f"\nKeep it up! You're scoring {mean_rewards} points against the opponent on average of 10 games with std dev {std_rewards} "
                                 f"but you are still {21-mean_rewards} points from winning the game. "
                                 f"Try improving paddle positioning to prevent opponent scoring.")
                elif mean_rewards <= 0:
                    feedback += (f"\nYour score is {mean_rewards} points on average of 10 games with std dev {std_rewards}. "
                                 f"Try to improve paddle positioning to prevent opponent scoring.")
                target = traj['observations'][-1]
                
                rewards.append(sum(traj['rewards']))
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}")
            policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))

            instruction = "In Pong, you control the right paddle and compete against the enemy on the left. "
            instruction += "The goal is to keep deflecting the ball away from your goal and into your enemy's goal to maximize your score and win the game by scoring close to 21 points. "
            instruction += "You score one point when the enemy misses the ball and the ball goes out of bounds on the enemy's side. "
            instruction += "The policy should move the paddle up or down or NOOP to hit the ball. If the paddle is below the ball, move the paddle up; otherwise, move the paddle down."
            instruction += "Analyze the trace to figure out the reason why you lose the game and optimize the code to score higher and higher points."
            
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
    finally:
        if env is not None:
            env.close()
    
    logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pong AI training")
    parser.add_argument("--env-name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n-optimization-steps", type=int, default=20)
    parser.add_argument("--memory-size", type=int, default=5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--sticky-action-p", type=float, default=0.0)
    args = parser.parse_args()

    frame_skip = args.frame_skip
    sticky_action_p = args.sticky_action_p
    env_name = args.env_name
    horizon = args.horizon
    n_optimization_steps = args.n_optimization_steps
    memory_size = args.memory_size

    # Create per-experiment directory
    experiment_dirs = create_experiment_dir("pong", timestamp)

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

    logger.info("Starting Pong AI training...")
    rewards = optimize_policy(
        env_name=env_name,
        horizon=horizon,
        n_optimization_steps=n_optimization_steps,
        memory_size=memory_size,
        verbose='output',
        frame_skip=frame_skip,
        sticky_action_p=sticky_action_p,
        logger=logger,
        experiment_dirs=experiment_dirs,
    )
    logger.info("Training completed.")
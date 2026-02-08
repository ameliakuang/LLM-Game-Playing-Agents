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
# from opto.optimizers import OptoPrime
from opto.optimizers.optoprime import OptoPrime
# from opto.optimizers.optoprime_v3 import (
#     OptoPrimeV3, ProblemInstance,
#     Content, OptimizerPromptSymbolSetJSON
# )
from trace_envs.breakout import TracedEnv
from logging_util import setup_logger
from training_utils import rollout, evaluate_policy, create_experiment_dir

gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        pre_ball_x = self.predict_ball_trajectory(obs)
        target_paddle_pos = self.generate_paddle_target(pre_ball_x, obs)
        action = self.select_paddle_action(target_paddle_pos, obs)
        return action

    @bundle(trainable=True)
    def predict_ball_trajectory(self, obs):
        """
        Predict the x-coordinate where the ball will intersect with the player's paddle by calculating its trajectory,
        using ball's (x, y) and (dx, dy) and accounting for bounces off the right and left walls.

        Game setup: 
        - Screen dimensions: The game screen has left and right walls and brick wall where the ball bounces 
          - Left wall: x=9
          - Right wall: x=152
        - Paddle positions:
          - Player paddle: bottom of screen (y=189)
        - Ball speed:
          - Ball deflects from higher-scoring bricks would have a higher speed and is harder to catch.
        - The paddle would deflect the ball at different angles depending on where the ball lands on the paddle
        
        Args:
            obs (dict): Dictionary containing object states for 'Player', 'Ball', and blocks '{color}B' (color in [R/O/Y/G/A/B]).
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            float: Predicted x-coordinate where the ball will intersect the player's paddle plane.
                  Returns None if ball position cannot be determined.
        """
        if 'Ball' not in obs:
            return None
            
    @bundle(trainable=True)
    def generate_paddle_target(self, pre_ball_x, obs):
        """
        Calculate the optimal x coordinate to move the paddle to catch the ball (at predicted_ball_x)
        and deflect the ball to hit bricks with higher scores in the brick wall.

        Logic:
        - Prioritize returning the ball when the ball is coming down (positive dy)
        - The brick wall consists of 6 vertically stacked rows from top to bottom:
          - Row 1 (top): Red bricks (7 pts)
          - Row 2: Orange (7 pts)
          - Row 3: Yellow (4 pts)
          - Row 4: Green (4 pts)
          - Row 5: Aqua (1 pt)
          - Row 6 (bottom): Blue (1 pt)
         - Strategic considerations:
          - Breaking lower bricks can create paths to reach higher-value bricks above
          - Creating vertical tunnels through the brick wall is valuable as it allows
            the ball to reach and bounce between high-scoring bricks at the top
          - Balance between safely returning the ball and creating/utilizing tunnels
            to access high-value bricks
        - Ball speed increases when hitting higher bricks, making it harder to catch

        Args:
            pre_ball_x (float): predicted x coordinate of the ball intersecting with the paddle or None
            obs (dict): Dictionary containing object states for 'Player', 'Ball', and blocks '{color}B' (color in [R/O/Y/G/A/B]).
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            float: Predicted x-coordinate to move the paddle to. 
                Returns None if ball position cannot be determined.
        """
        if pre_ball_x is None or 'Ball' not in obs:
            return None

        return None
        


    @bundle(trainable=True)
    def select_paddle_action(self, target_paddle_pos, obs):
        """
        Select the optimal action to move player paddle by comparing current player position and target_paddle_pos.

        Movement Logic:
        - If the player paddle's center position is GREATER than target_paddle_pos: Move LEFT (action 3)
        - If the player paddle's center position is LESS than target_paddle_pos: Move RIGHT (action 2)
        - If the player paddle is already aligned with target_paddle_pos: NOOP (action 0)
          (to stabilize the paddle when it's in position)
        Ensure stable movement to avoid missing the ball when close by.

        Args:
            target_paddle_pos (float): predicted x coordinate of the position to best position the paddle to catch the ball,
                and hit the ball to break brick wall.
            obs (dict): Dictionary containing object states for 'Player', 'Ball', and blocks '{color}B' (color in [R/O/Y/G/A/B]).
                Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            int: 0 for NOOP, 2 for RIGHT, 3 for LEFT
        """
        if target_paddle_pos is None or 'Player' not in obs:
            return 0
            
        paddle = obs['Player']
        paddle_x = paddle['x']
        paddle_w = paddle['w']
        paddle_center = paddle_x + (paddle_w / 2)
        
        # Add deadzone to avoid oscillation
        deadzone = 2
        if abs(paddle_center - target_paddle_pos) < deadzone:
            return 0  # NOOP if close enough
        elif paddle_center > target_paddle_pos:
            return 3  # LEFT
        else:
            return 2  # RIGHT




def optimize_policy(
    env_name="BreakoutNoFrameskip-v4",
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
        experiment_dirs = create_experiment_dir("breakout", timestamp)

    policy = Policy()
    if policy_ckpt:
        logger.info(f"Continuing training from ckpt: {policy_ckpt}")
        policy.load(policy_ckpt)
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size, max_tokens=4096)
    env = TracedEnv(env_name=env_name,
                    frameskip=frame_skip,
                    repeat_action_probability=sticky_action_p,)
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
                if mean_rewards >= 350:
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
                if mean_rewards >= 300:
                    feedback += (f"\nGood job! You're close to winning the game! "
                                 f"You're scoring {mean_rewards} points against the opponent on average of {num_episodes} games with std dev {std_rewards}, "
                                 f"try ensuring you return the ball, "
                                 f"only {350-mean_rewards} points short of winning.")
                elif mean_rewards > 0:
                    feedback += (f"\nKeep it up! You're scoring {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards} "
                                 f"but you are still {350-mean_rewards} points from winning the game. "
                                 f"Try improving paddle positioning to return the ball and avoid losing lives.")
                elif mean_rewards <= 0:
                    feedback += (f"\nYour score is {mean_rewards} points on average of {num_episodes} games with std dev {std_rewards}. "
                                 f"Try to improve paddle positioning to return the ball and avoid losing lives.")
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
            

            instruction = "In Breakout, you move the bottom paddle left and right to deflect the ball to a top block wall. "
            instruction += "The goal is to keep deflecting the ball to the block wall to destroy the bricks upon contact and score points. "
            instruction += "The brick wall consists of six rows of different colored bricks, each worth different points when hit: "
            instruction += "Red: top row, 7 pts; Orange: 2nd row: 7 pts, Yellow: 3rd row: 4 pts, Green: 4th row: 4 pts, Aqua: 5th row: 1 pt, Blue: 6th row: 1pt. "
            instruction += "The game screen has left and right walls and brick wall where the ball bounces: left wall: x=9, right wall: x=152, player paddle: y=189. "
            instruction += "Hitting higher bricks would deflect the ball faster and make catching the ball harder. "
            instruction += "You will win the game when you score >= 400 points. "
            instruction += "You lose a life when you fail to catch the ball and the ball moves below the paddle. The game ends when you lose 5 lives. "
            instruction += "Analyze the trace to figure out the reason why you lose the game and optimize the code to score higher points by prioritizing hitting higher-value bricks when possible while maintaining ball control."
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
    
    logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    frame_skip = 4
    sticky_action_p = 0.0
    env_name = "BreakoutNoFrameskip-v4"
    horizon = 300
    n_optimization_steps = 30
    memory_size = 5
    policy_ckpt = None

    # Create per-experiment directory
    experiment_dirs = create_experiment_dir("breakout", timestamp)

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

    logger.info("Starting Breakout AI training...")
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

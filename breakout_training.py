import os
import ale_py
import logging
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
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari
gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
base_trace_ckpt_dir = Path("trace_ckpt")
base_trace_ckpt_dir.mkdir(exist_ok=True)


class TracedEnv:
    def __init__(self, 
                 env_name="BreakoutNoFrameskip-v4",
                 render_mode="human",
                 obs_mode="ori",
                 hud=False,
                 frameskip=4,
                 repeat_action_probability=0.0):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.hud = hud
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.env = None
        self.current_lives = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = OCAtari(self.env_name, 
                    render_mode=self.render_mode, 
                    obs_mode=self.obs_mode, 
                    hud=self.hud,
                    frameskip=self.frameskip,
                    repeat_action_probability=self.repeat_action_probability)
        self.obs, _ = self.env.reset()
        self.current_lives = self.env._env.unwrapped.ale.lives()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
            self.current_lives = None
    
    def __del__(self):
        self.close()

    def extract_game_state(self, objects, rgb, info):
        obs = dict()
        color_blocks = {
            "Red": [], "Orange": [], "Yellow": [],
            "Green": [], "Aqua": [], "Blue": []
        }
        for object in objects:
            if object.category == "NoObject":
                continue
            elif object.category == "Block":
                color = None
                if object.y == 57: color = "Red"
                elif object.y == 63: color = "Orange"
                elif object.y == 69: color = "Yellow"
                elif object.y == 75: color = "Green"
                elif object.y == 81: color = "Aqua"
                elif object.y == 87: color = "Blue"
                else: continue  # Skip unknown y-positions
            
                color_blocks[color].append({
                    "x": object.x,
                    "y": object.y,
                    "w": object.w,
                    "h": object.h,
                })
            else:
                obs[object.category] = {"x": object.x,
                                        "y": object.y,
                                        "w": object.w,
                                        "h": object.h,
                                        "dx": object.dx,
                                        "dy": object.dy,}
        for color, blocks in color_blocks.items():
            if blocks:
                obs[f"{color[0]}B"] = blocks
        if info:
            obs['lives'] = info.get('lives', None)
        return obs


    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        _, _ = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(1)
        self.current_lives = info.get('lives')
        self.obs = self.extract_game_state(self.env.objects, obs, info)
        self.obs['reward'] = np.nan

        return self.obs, info
    
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            lives = info.get('lives')
            if self.current_lives and lives < self.current_lives:
                next_obs, reward, termination, truncation, info = self.env.step(1)
            self.current_lives = lives

            self.obs = self.extract_game_state(self.env.objects, next_obs, info)
            self.obs['reward'] = reward
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)
        @bundle()
        def step(action):
            """
            Take action in the environment and return the next observation
            """
            return self.obs

        self.obs = step(action)
        return self.obs, reward, termination, truncation, info

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
            obs (dict): Dictionary containing object states for "Player", "Ball", and blocks "{color}B" (color in [R/O/Y/G/A/B]).
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
            obs (dict): Dictionary containing object states for "Player", "Ball", and blocks "{color}B" (color in [R/O/Y/G/A/B]).
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
            obs (dict): Dictionary containing object states for "Player", "Ball", and blocks "{color}B" (color in [R/O/Y/G/A/B]).
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



def rollout(env, horizon, policy):
    """Rollout a policy in an env for horizon steps."""
    try:
        obs, _ = env.reset()
        trajectory = dict(observations=[], actions=[], rewards=[], terminations=[], truncations=[], infos=[], steps=0)
        trajectory["observations"].append(obs)
        
        for i in range(horizon):
            error = None
            try:
                action = policy(obs)
                next_obs, reward, termination, truncation, info = env.step(action)
            except trace.ExecutionError as e:
                error = e
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            
            if error is None:
                trajectory["observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminations"].append(termination)
                trajectory["truncations"].append(truncation)
                trajectory["infos"].append(info)
                trajectory["steps"] += 1
                if termination or truncation:
                    break
                obs = next_obs
    finally:
        env.close()
    
    return trajectory, error

def test_policy(policy, 
                num_episodes=1, 
                steps_per_episode=4000,
                frameskip=1,
                repeat_action_probability=0.0,
                logger=None):
    logger.info("Evaluating policy")
    env = TracedEnv(render_mode=None,
                    frameskip=frameskip,
                    repeat_action_probability=repeat_action_probability)
    rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

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
):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    policy = Policy()
    if policy_ckpt:
        logger.info(f"Continuing training from ckpt: {policy_ckpt}")
        policy.load(policy_ckpt)
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    env = TracedEnv(env_name=env_name,
                    frameskip=frame_skip,
                    repeat_action_probability=sticky_action_p,)
    perf_csv_filename = log_dir / f"perf_{env_name.replace("/", "_")}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}.csv"
    trace_ckpt_dir = base_trace_ckpt_dir / f"{env_name.replace("/", "_")}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}"
    trace_ckpt_dir.mkdir(exist_ok=True)
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
                mean_rewards, std_rewards = test_policy(policy,
                                                        num_episodes=num_episodes,
                                                        steps_per_episode=steps_per_episode,
                                                        frameskip=frame_skip,
                                                        repeat_action_probability=sticky_action_p,
                                                        logger=logger) # run the policy on 10 games of length 4000 steps each
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
    frame_skip = 4
    sticky_action_p = 0.0
    env_name = "BreakoutNoFrameskip-v4"
    horizon = 300
    n_optimization_steps = 30
    memory_size = 5
    policy_ckpt = None

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    # Set up file logging
    log_file = log_dir / f"{env_name.replace('/', '_')}_OCAtari_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
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

    )
    logger.info("Training completed.")

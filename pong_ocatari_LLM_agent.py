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


from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym 
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari

load_dotenv(override=True)
gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
base_trace_ckpt_dir = Path("trace_ckpt")
base_trace_ckpt_dir.mkdir(exist_ok=True)

class PongOCAtariTracedEnv:
    def __init__(self, 
                 env_name="PongNoFrameskip-v4",
                 render_mode="human",
                 obs_mode="obj",
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
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
    
    def __del__(self):
        self.close()

    def extract_obj_state(self, objects):
        obs = dict()
        for object in objects:
            obs[object.category] = {"x": object.x,
                                    "y": object.y,
                                    "w": object.w,
                                    "h": object.h,
                                    "dx": object.dx,
                                    "dy": object.dy,}
        return obs


    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        _, info = self.env.reset()
        self.obs = self.extract_obj_state(self.env.objects)
        self.obs['reward'] = np.nan

        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = self.extract_obj_state(self.env.objects)
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
          - Top boundary: approximately y=30
          - Bottom boundary: approximately y=190
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
            


def rollout(env, horizon, policy):
    """Rollout a policy in an env for horizon steps."""
    try:
        obs, _ = env.reset()
        trajectory = dict(observations=[], actions=[], rewards=[], terminations=[], truncations=[], infos=[], steps=0)
        trajectory["observations"].append(obs)
        
        for _ in range(horizon):
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
                num_episodes=10, 
                steps_per_episode=4000,
                frameskip=1,
                repeat_action_probability=0.0):
    logger.info("Evaluating policy")
    env = PongOCAtariTracedEnv(render_mode=None,
                               frameskip=frameskip,
                               repeat_action_probability=repeat_action_probability)
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
        
        rewards.append(episode_reward)
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

def optimize_policy(
    env_name="PongNoFrameskip-v4",
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    frame_skip=4,
    sticky_action_p=0.00,
    logger=None,
    # model="gpt-4o-mini"
):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get the config file path from environment variable
    # config_path = os.getenv("OAI_CONFIG_LIST")
    # config_list = config_list_from_json(config_path)
    # config_list = [config for config in config_list if config["model"] == model]
    # optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)

    policy = Policy()
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    env = PongOCAtariTracedEnv(env_name=env_name,
                               frameskip=frame_skip,
                               repeat_action_probability=sticky_action_p)
    perf_csv_filename = log_dir / f"perf_{env_name.replace("/", "_")}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}.csv"
    trace_ckpt_dir = base_trace_ckpt_dir / f"{env_name.replace("/", "_")}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}"
    trace_ckpt_dir.mkdir(exist_ok=True)
    try:
        rewards = []
        optimization_data = []
        logger.info("Optimization Starts")
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                mean_rewards, std_rewards = test_policy(policy,
                                                        frameskip=frame_skip,
                                                        repeat_action_probability=sticky_action_p) # run the policy on 10 games of length 4000 steps each
                if mean_rewards >= 21:
                    logger.info("Congratulations! You've achieved a perfect score of {mean_rewards} with std dev {std_rewards}. Ending optimization early.")
                    rewards.append(sum(traj['rewards']))
                    optimization_data.append({
                        "Optimization Step": i,
                        "Mean Reward": mean_rewards,
                        "Std Dev Reward": std_rewards
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
                optimization_data.append({
                    "Optimization Step": i,
                    "Mean Reward": mean_rewards,
                    "Std Dev Reward": std_rewards
                })
                df = pd.DataFrame(optimization_data)
                df.to_csv(perf_csv_filename, index=False)
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
    finally:
        if env is not None:
            env.close()
    
    logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    frame_skip = 4
    sticky_action_p = 0.0
    env_name = "PongNoFrameskip-v4"
    horizon = 400
    n_optimization_steps = 20
    memory_size = 5

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
        # model="gpt-4o-mini"

    )
    logger.info("Training completed.")
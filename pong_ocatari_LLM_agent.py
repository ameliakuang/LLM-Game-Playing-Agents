import os
import ale_py
import cv2
import logging
import datetime
import sys
from pathlib import Path
import numpy as np
import io
import contextlib
from copy import copy

from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari

load_dotenv()
gym.register_envs(ale_py)

class PongOCAtariTracedEnv:
    def __init__(self, 
                 env_name="ALE/Pong-v5",
                 render_mode="human",
                 obs_mode="obj",
                 hud=False):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.hud = hud
        self.env = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = OCAtari(self.env_name, 
                           render_mode=self.render_mode, 
                           obs_mode=self.obs_mode, 
                           hud=self.hud)
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
        Given current observation, compute the velocity of the ball and predict the y coordinate of the ball when it reaches the player's paddle. 
        
        Args:
            obs (dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
            
        Returns:
            float: The predicted y-coordinate where the ball will reach the player's side
        """
        if 'Ball' in obs:
            ball_y = obs['Ball'].get('y')
            ball_dy = obs['Ball'].get('dy', 0)
            ball_x = obs['Ball'].get('x', 0)
            player_x = obs['Player'].get('x', 0)
            ball_dx = obs['Ball'].get('dx', 0)

            if ball_y is not None and ball_dy is not None:
                if ball_dx == 0:  # Considering if ball is not moving horizontally
                    return ball_y
                frames_to_reach = (player_x - ball_x) / max(1, ball_dx)  # Ensure non-zero division
                predicted_y = ball_y + ball_dy * frames_to_reach
                return predicted_y
        return None
    
    @bundle(trainable=True)
    def select_action(self, predicted_ball_y, obs):
        '''
        Select the optimal action of the player paddle comparing current player position and predicted ball y coordinate.

        Args:
            predicted_ball_y (float): predicted y coordinate of the ball or None
            obs(dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
        Returns:
            int: 0 for NOOP, 2 for DOWN, 3 for UP
        '''

        if predicted_ball_y is not None and "Player" in obs:
            player_y = obs["Player"].get("y", None)
            player_height = obs["Player"].get("h", None)

            if player_y is not None and player_height is not None:
                player_center = player_y + player_height / 2
                if player_center < predicted_ball_y:
                    return 3
                elif player_center > predicted_ball_y:
                    return 2
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

def test_policy(policy, num_episodes=10, steps_per_episode=4000):
    logger.info("Evaluating policy")
    env = PongOCAtariTracedEnv(render_mode=None)
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
    env_name="ALE/Pong-v5",
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    model="gpt-4o-mini"
):
    
    
    # Get the config file path from environment variable
    # config_path = os.getenv("OAI_CONFIG_LIST")
    # config_list = config_list_from_json(config_path)
    # config_list = [config for config in config_list if config["model"] == model]
    # optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)

    policy = Policy()

    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    
    env = PongOCAtariTracedEnv(env_name=env_name)
    try:
        rewards = []
        logger.info("Optimization Starts")
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                mean_rewards, std_rewards = test_policy(policy)
                if mean_rewards >= 19:
                    feedback += f"\nGood job! You're close to winning the game!"
                if mean_rewards > 0:
                    feedback += f"\nKeep it up! You're scoring {mean_rewards} points against the opponent on average of 10 games with std dev {std_rewards} but you are still {21-mean_rewards} points from winning the game. Try improving paddle positioning to prevent opponent scoring."
                elif mean_rewards <= 0:
                    feedback += f"\nYour score is {mean_rewards} points on average of 10 games with std dev {std_rewards}. Try to improve paddle positioning to prevent opponent scoring."
                target = traj['observations'][-1]
                
                rewards.append(sum(traj['rewards']))
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {policy.parameters()}")

            instruction = "In Pong, you control the right paddle and compete against the computer on the left. "
            instruction += "The goal is to keep deflecting the ball away from your goal and into your opponent's goal to maximize your score and win the game by scoring close to 21 points. "
            instruction += "You score one point when the opponent misses the ball or hits it out of bounds. "
            instruction += "The policy should move the paddle up or down or NOOP to hit the ball. If the paddle is below the ball, move the paddle up; otherwise, move the paddle down."
            
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
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, Parameter: {policy.parameters()}")
    finally:
        if env is not None:
            env.close()
    
    logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Set up file logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pong_ai_OCAtari_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Pong AI training...")
    rewards = optimize_policy(
        env_name="ALE/Pong-v5",
        horizon=400,
        n_optimization_steps=20,
        memory_size=5,
        verbose='output',
        model="gpt-4o-mini"
    )
    logger.info("Training completed.")
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


load_dotenv()
gym.register_envs(ale_py)

def process_image(obs):
    """Process the grayscale image into contours.
    
    Args:
        obs: Grayscale image of the game screen
        
    Returns:
        dict: Dictionary containing position of ball, agent paddle and opponent paddle found in the image in [x, y, w, h] format
    """
    # Crop relevant part of the frame (excluding scores and borders)
    gray = obs[34:194, 15:147]
    
    # Threshold to separate objects (ball and paddles are bright)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Visualize the contours
    # contour_image = gray.copy()
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("Contours", contour_image)
    # cv2.waitKey(1)

    ball_contour = None
    paddle_contour = None
    opponent_contour = None
        
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Ball is typically a small square
        if area < 5:
            ball_contour = [x, y, w, h]
        # Agent paddle is on the right side
        elif x > 100:
            paddle_contour = [x, y, w, h]
        else:
            opponent_contour = [x, y, w, h]
    
    return {"ball_pos": ball_contour, "paddle_pos": paddle_contour, "opponent_pos": opponent_contour}

class PongTracedEnv:
    def __init__(self, 
                 env_name="ALE/Pong-v5",
                 render_mode="human",
                 obs_type="grayscale"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.env = None
        self.prev_obs = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = gym.make(self.env_name, render_mode=self.render_mode, obs_type=self.obs_type)
        self.env.reset()
        self.obs = None
        self.prev_obs = None
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()
    
    def _add_prefix_to_keys(self, input_dict, prefix):
        """Adds a prefix to the keys of a dictionary.
        """
        return {f"{prefix}{key}": value for key, value in input_dict.items()}

    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset()
        self.obs = process_image(obs)
        self.obs['reward'] = np.nan
        self.prev_obs = {
            'ball_pos': None,
            'paddle_pos': None,
            'opponent_pos': None,
            'reward': np.nan
        }
        self.obs.update(self._add_prefix_to_keys(self.prev_obs, "prev_"))

        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            current_obs = {
                'ball_pos': self.obs.get('ball_pos', None),
                'paddle_pos': self.obs.get('paddle_pos', None),
                'opponent_pos': self.obs.get('opponent_pos', None),
                'reward': self.obs.get('reward', np.nan)
            }
            self.prev_obs = current_obs
            self.obs = next_obs = process_image(next_obs)
            self.obs['reward'] = next_obs['reward'] = reward
            if self.prev_obs:
                self.obs.update(self._add_prefix_to_keys(self.prev_obs, "prev_"))

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
            return next_obs

        next_obs = step(action)
        return next_obs, reward, termination, truncation, info

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

def optimize_policy(
    env_name="ALE/Pong-v5",
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    model="gpt-4o-mini"
):
    @trace.bundle(trainable=True)
    def policy(obs):
        '''
        A policy that moves the paddle up or down to catch and deflect a moving ball.
        IMPORTANT!! If the paddle is BELOW the ball, move the paddle UP; otherwise, move the paddle DOWN.
        Make prediction on the ball's moving direction and velocity to adjust the paddle action.
        If the ball is off-screen, then ball_pos would be None and there's no need to adjust the paddle (NOOP).

        Args:
            obs (dict): A dictionary with keys ("ball_pos", "paddle_pos", "prev_ball_pos", "prev_paddle_pos") and values the corresponding [x, y, w, h], coordinates, width and height of the ball and agent paddle in the screen.
        Output:
            action (int): The action to take among 0 (NOOP), 1 (FIRE), 2 (DOWN), 3 (UP).
        '''
        ball_pos = obs.get('ball_pos', None)
        prev_ball_pos = obs.get('prev_ball_pos', None)
        paddle_pos = obs.get('paddle_pos', None)
        
        if ball_pos is None or paddle_pos is None:
            return 0  # NOOP if no ball or paddle
        
        # Ball's vertical center
        ball_y = ball_pos[1] + ball_pos[3] // 2
        
        # Paddle's vertical center and dimensions
        paddle_y = paddle_pos[1] + paddle_pos[3] // 2
        paddle_height = paddle_pos[3]
        
        # Predict ball movement if previous position exists
        predicted_ball_y = ball_y
        if prev_ball_pos is not None:
            prev_ball_center = prev_ball_pos[1] + prev_ball_pos[3] // 2
            ball_velocity = ball_y - prev_ball_center
            predicted_ball_y += ball_velocity
        
        # Create a small buffer zone around paddle center
        upper_buffer = paddle_y - paddle_height // 4
        lower_buffer = paddle_y + paddle_height // 4
        
        # More nuanced positioning
        if predicted_ball_y < upper_buffer:
            return 3  # Move UP
        elif predicted_ball_y > lower_buffer:
            return 2  # Move DOWN
        else:
            return 0  # NOOP if ball is within paddle's center zone
    
    # Get the config file path from environment variable
    # config_path = os.getenv("OAI_CONFIG_LIST")
    # config_list = config_list_from_json(config_path)
    # config_list = [config for config in config_list if config["model"] == model]
    # optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    
    env = PongTracedEnv(env_name=env_name)
    try:
        rewards = []
        logger.info("Optimization Starts")
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                if sum(traj['rewards']) > 0:
                    feedback += "\nGood job! You're scoring points against the opponent."
                elif sum(traj['rewards']) <= 0:
                    feedback += "\nTry to improve paddle positioning to prevent opponent scoring."
                target = traj['observations'][-1]
                
                rewards.append(sum(traj['rewards']))
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {policy.parameters()}")

            instruction = "In Pong, you control the right paddle and compete against the computer on the left. "
            instruction += "The goal is to keep deflecting the ball away from your goal and into your opponent's goal to maximize your score and win the game. "
            instruction += "You score one point when the opponent misses the ball or hits it out of bounds. "
            instruction += "The policy should move the paddle up or down or NOOP to hit the ball. If the paddle is below the ball, move the paddle up; otherwise, move the paddle down."
            
            optimizer.objective = instruction + optimizer.default_objective
            
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
    log_file = log_dir / f"pong_ai_{timestamp}.log"
    
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
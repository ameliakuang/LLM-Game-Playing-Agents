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

from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
import random

load_dotenv()
gym.register_envs(ale_py)

def process_frame(gray):
    """
    Processes a grayscale frame from Space Invaders.
    
    This function thresholds the frame to extract game objects,
    then uses contour detection to estimate the positions of the player's spaceship,
    enemy invaders, and enemy lasers.
    """
    # Threshold to highlight bright objects (spaceship, invaders, lasers, etc.)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Visualize the thresholded image
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(1)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize the detected contours
    contour_image = gray.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(1)
    
    spaceship_pos = None
    invaders = []
    lasers = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Heuristic: the player's spaceship is usually larger and located near the bottom
        if y > 150 and area > 10:
            spaceship_pos = (x + w // 2, y + h // 2)
        # Heuristic: enemy invaders are typically small and appear in the upper half
        elif y < 100 and 2 < area < 50:
            invaders.append((x + w // 2, y + h // 2))
        # Heuristic: enemy lasers might appear in the mid-region (between invaders and spaceship)
        # and have a relatively small area.
        elif 100 <= y <= 150 and 5 < area < 15:
            lasers.append((x + w // 2, y + h // 2))
    
    return spaceship_pos, invaders, lasers

@bundle(trainable=True)
def decide_movement(spaceship_pos, invaders, lasers, laser_margin=10, chase_margin=30):
    # 2: move right, 3: move left, 0: no movement

    # Check for laser threat.
    
    # If no immediate laser threat, check if chasing invaders is necessary.
    
    return 0 if random.random() < 0.5 else 2  # Dummy agent: no movement or move right

@bundle(trainable=True)
def decide_firing():
    """
    Firing decision: continuously fire.
    
    Returns:
        1: FIRE action
    """
    return 0 if random.random() < 0.8 else 1

def space_invaders_policy(observation):
    """
    Heuristic policy for Space Invaders.
    
    It uses two independent trainable functions:
      - decide_movement: for movement based on laser avoidance and chasing invaders.
      - decide_firing: for continuously firing.
      
    The high-level decision logic:
      1. If the movement function indicates a movement (non-0 action), that action is chosen.
      2. Otherwise, the agent fires.
    """
    spaceship_pos, invaders, lasers = process_frame(observation)
    action = 0  # Default: NOOP
    
    if spaceship_pos:
        move_action = decide_movement(spaceship_pos, invaders, lasers)
        if move_action != 0:
            print(f"\rMovement: {move_action} (laser avoidance or chasing), Spaceship: {spaceship_pos}", end="")
            action = move_action
        else:
            print(f"\rFiring continuously. Spaceship: {spaceship_pos}", end="")
            action = decide_firing()
    else:
        print("\rNo detection - Spaceship not found", end="")
    
    return action

class SpaceInvadersTracedEnv:
    def __init__(self, 
                 env_name="ALE/SpaceInvaders-v5",
                 render_mode="human",
                 obs_type="grayscale"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.env = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = gym.make(self.env_name, render_mode=self.render_mode, obs_type=self.obs_type)
        self.env.reset()
        self.obs = None
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()
    
    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset()
        spaceship_pos, invaders, lasers = process_frame(obs)
        self.obs = {"spaceship_pos": spaceship_pos, "invaders": invaders, "lasers": lasers, "reward": np.nan}
        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            spaceship_pos, invaders, lasers = process_frame(next_obs)
            self.obs = {"spaceship_pos": spaceship_pos, "invaders": invaders, "lasers": lasers, "reward": reward}
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)
        
        @bundle()
        def step_bundle(action):
            return self.obs
        
        next_obs = step_bundle(action)
        return next_obs, reward, termination, truncation, info

def rollout(env, horizon, policy):
    """Rollout a policy in an environment for a given horizon."""
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
    env_name="ALE/SpaceInvaders-v5",
    horizon=800,
    memory_size=5,
    n_optimization_steps=5,
    verbose=False,
    model="gpt-4o-mini"
):
    def policy(observation):
        """
        High-level policy that decides between movement and firing.
        """
        if hasattr(observation, "data"):
            observation = observation.data
        spaceship_pos = observation.get("spaceship_pos")
        invaders = observation.get("invaders")
        lasers = observation.get("lasers")
        action = 0  # Default action: NOOP
        
        if spaceship_pos:
            move_action = decide_movement(spaceship_pos, invaders, lasers)
            if move_action != 0:
                print(f"\rMovement: {move_action} (laser avoidance or chasing), Spaceship: {spaceship_pos}", end="")
                action = move_action
            else:
                print(f"\rFiring continuously. Spaceship: {spaceship_pos}", end="")
                action = decide_firing()
        else:
            print("\rNo detection - Spaceship not found", end="")
        
        return action
    
    # Retrieve configuration list for the optimizer.
    config_path = os.getenv("OAI_CONFIG_LIST")
    config_list = config_list_from_json(config_path)
    config_list = [config for config in config_list if config["model"] == model]
    # Combine parameters from the trainable functions.
    parameters = list(decide_movement.parameters()) + list(decide_firing.parameters())
    optimizer = OptoPrime(parameters, config_list=config_list, memory_size=memory_size)
    
    env = SpaceInvadersTracedEnv(env_name=env_name)
    logger.info("Optimization Starts")
    rewards = []
    try:
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                total_reward = sum(traj['rewards'])
                feedback = f"Episode ends after {traj['steps']} steps with total score: {total_reward:.1f}"
                if total_reward > 0:
                    feedback += "\nGood job! Enemies are being hit."
                else:
                    feedback += "\nTry to improve movement and firing strategy."
                target = traj['observations'][-1]
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {parameters}")

            instruction = (
                "In Space Invaders, you control a spaceship at the bottom of the screen. "
                "Avoid enemy lasers and chase enemy invaders to shoot them. "
                "Continuously fire if no movement is needed. "
            )
            optimizer.objective = instruction + optimizer.default_objective
            
            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))
            
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=verbose)
                llm_output = stdout_buffer.getvalue()
                if llm_output:
                    logger.info(f"LLM response:\n{llm_output}")
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, Parameter: {parameters}")
            rewards.append(total_reward)
    finally:
        if env is not None:
            env.close()
    
    avg_reward = sum(rewards) / len(rewards) if rewards else float('nan')
    logger.info(f"Final Average Reward: {avg_reward}")
    return rewards

if __name__ == "__main__":
    # Set up logging to console and file.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Set up file logging.
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"space_invaders_ai_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Space Invaders AI training...")
    rewards = optimize_policy(
        env_name="ALE/SpaceInvaders-v5",
        horizon=800,
        n_optimization_steps=5,
        memory_size=5,
        verbose='output',
        model="gpt-4o-mini"
    )
    logger.info("Training completed.")

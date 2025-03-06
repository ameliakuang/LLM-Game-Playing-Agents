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

load_dotenv()
gym.register_envs(ale_py)

def process_image(obs):
    """
    Process the grayscale image from Space Invaders to extract game objects.

    Args:
        obs: Grayscale image of the game screen.

    Returns:
        dict: Dictionary containing positions of the player's spaceship, enemy invaders, and enemy lasers.
              - "spaceship_pos": tuple (x, y) for the spaceship center.
              - "invaders": list of tuples (x, y) for enemy invaders.
              - "lasers": list of tuples (x, y) for enemy lasers.
    """
    # Threshold to highlight bright objects (spaceship, invaders, lasers, etc.)
    _, thresh = cv2.threshold(obs, 100, 255, cv2.THRESH_BINARY)

    # Optional visualization (commented out for optimization)
    # cv2.imshow("Threshold", thresh)
    # cv2.waitKey(1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Optional visualization of contours (commented out)
    # contour_image = obs.copy()
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("Contours", contour_image)
    # cv2.waitKey(1)

    spaceship_pos = None
    invaders = []
    lasers = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Heuristic: the player's spaceship is usually larger and near the bottom
        if y > 150 and area > 10:
            spaceship_pos = (x + w // 2, y + h // 2)
        # Heuristic: enemy invaders are typically small and appear in the upper half
        elif y < 100 and 2 < area < 50:
            invaders.append((x + w // 2, y + h // 2))
        # Heuristic: enemy lasers appear in the mid-region and have a small area
        elif 100 <= y <= 150 and 5 < area < 15:
            lasers.append((x + w // 2, y + h // 2))

    return {"spaceship_pos": spaceship_pos, "invaders": invaders, "lasers": lasers}

class SpaceInvadersTracedEnv:
    def __init__(self, env_name="ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale"):
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
        self.obs = process_image(obs)
        self.obs['reward'] = np.nan
        return self.obs, info

    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = next_obs = process_image(next_obs)
            self.obs['reward'] = next_obs['reward'] = reward
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
            """
            Take action in the environment and return the next observation.
            """
            return next_obs

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
    horizon=2000,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    model="gpt-4o-mini"
):
    @trace.bundle(trainable=True)
    def policy(obs):
        '''
        A policy that controls the spaceship in Space Invaders.
        
        The policy processes the observation to detect the spaceship, enemy invaders, and enemy lasers.
        It then makes decisions based on the following heuristics:
            - If spaceship_pos not detected, directly return 0 (NOOP)
            - If an enemy laser is detected within a certain horizontal margin of the spaceship, move right (action 2) to avoid it. If the spaceship can't move right (i.e. near the wall), move left
            - If no immediate laser threat exists and enemy invaders are present, compute the average x-position of invaders:
                • If the spaceship is left of (average - chase_margin), move right (action 2).
                • If the spaceship is right of (average + chase_margin), move left (action 3).
            - If neither condition is met, fire (action 1).
        
        Returns:
            action (int): The selected action among 0 (NOOP), 1 (FIRE), 2 (move RIGHT), 3 (move LEFT).
        '''
        spaceship_pos = obs["spaceship_pos"]
        invaders = obs["invaders"]
        lasers = obs["lasers"]

        action = 0  # Default NOOP
        laser_margin = 5
        chase_margin = 60
        right_margin = 30
        if not spaceship_pos:
            return 0 

        return 0

    # Get the config file path from environment variable
    config_path = os.getenv("OAI_CONFIG_LIST")
    config_list = config_list_from_json(config_path)
    config_list = [config for config in config_list if config["model"] == model]
    optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)

    env = SpaceInvadersTracedEnv(env_name=env_name)
    try:
        rewards = []
        logger.info("Optimization Starts")
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                feedback = f"Episode ends after {traj['steps']} steps with total score: {sum(traj['rewards']):.1f}"
                if sum(traj['rewards']) > 0:
                    feedback += "\nGood job! You're scoring points against the invaders."
                elif sum(traj['rewards']) <= 0:
                    feedback += "\nTry to improve spaceship positioning and timing to avoid enemy lasers."
                target = traj['observations'][-1]
                rewards.append(sum(traj['rewards']))
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node

            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {policy.parameters()}")

            instruction = "In Space Invaders, you control the player's spaceship and fend off waves of enemy invaders. "
            instruction += "Avoid enemy lasers and move horizontally to align with the invaders. "
            instruction += "Fire at the enemies to score points. "
            instruction += "The policy should decide whether to move left, move right, fire, or do nothing (NOOP). "

            optimizer.objective = instruction + optimizer.default_objective

            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=False)
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
    log_file = log_dir / f"space_invader_ai_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Starting Space Invaders AI training...")
    rewards = optimize_policy(
        env_name="ALE/SpaceInvaders-v5",
        horizon=400,
        n_optimization_steps=5,
        memory_size=5,
        verbose='output',
        model="gpt-4o-mini"
    )
    logger.info("Training completed.")

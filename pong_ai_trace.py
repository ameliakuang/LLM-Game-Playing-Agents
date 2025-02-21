from dotenv import load_dotenv
from autogen import config_list_from_json
import random
import numpy as np
import os
import gymnasium as gym
import ale_py
import cv2
import opto.trace as trace
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError

# from simple_pong_ai import process_frame


load_dotenv()
# config_list = config_list_from_json("OAI_CONFIG.json")
# print(config_list)

gym.register_envs(ale_py)

class PongTracedEnv:
    def __init__(self, 
                 env_name="ALE/Pong-v5",
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
    
    @trace.bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset()
        self.obs = process_obs(obs)
        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = next_obs = process_obs(next_obs)
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)

        @trace.bundle()
        def step(action):
            """
            Take action in the environment and return the next observation
            """
            return next_obs

        next_obs = step(action)
        return next_obs, reward, termination, truncation, info

def process_obs(obs):
    '''Process the grayscale image of game screen into a dictionary of ball and paddle positions'''
    # Crop relevant part of the frame (excluding scores and borders)
    gray = obs[34:194, 15:147]  # Crop game area
    
    # Threshold to separate objects (ball and paddles are bright)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Visualize the thresholding
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(1)
    
    # Find contours of objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize the contours
    contour_image = gray.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(1)
    
    ball_pos = None
    paddle_pos = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Ball is typically a small square
        if area < 5:
            ball_pos = [x, y]
        # Agent paddle is on the right side
        elif x > 100:  # Right paddle (agent)
            paddle_pos = [x, y]
    
    return {"ball_pos": ball_pos, "paddle_pos": paddle_pos}

# class Policy(Module):
#     def init(self):
#         pass

#     def __call__(self, obs):
#         return 
    
#     @trace.bundle(trainable=True)
#     def reason(self, obs):
#         pass

#     @trace.bundle(trainable=True)
#     def act(self, action):
#         pass

def rollout(env, horizon, policy):
    """Rollout a policy in an env for horizon steps."""
    traj = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminations": [],
        "truncations": [],
        "steps": 0,
        "total_reward": 0.0  # Track total reward
    }
    obs, info = env.reset()
    traj["observations"].append(obs)

    for _ in range(horizon):
        error = None
        try:
            action = policy(obs)
            next_obs, reward, termination, truncation, info = env.step(action)
        except trace.ExecutionError as e:
            error = e
            break
        if not error:
            traj["observations"].append(next_obs)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["terminations"].append(termination)
            traj["truncations"].append(truncation)
            traj["total_reward"] += reward  # Accumulate reward
            traj["steps"] += 1
            if termination or truncation:
                env.reset()
                break
            obs = next_obs
    env.close()
    return traj, error


def optimize_policy(
    horizon,
    env_name="ALE/Pong-v5",
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    model="gpt-4o-mini"
):
    @trace.bundle(trainable=True)
    def policy(obs):
        """ Heuristic policy: Moves paddle towards the ball. 
        
        Args:
            obs (dictionary): The current observation from the environment, containing ball_pos (x, y) coordinate of the ball and paddle_pos (x, y) coordinate of the agent paddle.
        
        Returns:
            action (int): The action to take among 0 (NOOP), 1 (FIRE), 2 (RIGHT), 3 (LEFT), 4 (RIGHTFIRE), 5(LEFTFIRE).
        """
        ball_pos, paddle_pos = obs['ball_pos'], obs['paddle_pos']
        
        action = 0  # NOOP
        
        if ball_pos and paddle_pos:
            ball_y = ball_pos[1]
            paddle_y = paddle_pos[1]
            
            if paddle_y + 10 < ball_y:  # Paddle is below the ball, move up
                action = 3  
            elif paddle_y > ball_y + 10:  # Paddle is above the ball, move down
                action = 2
        return action
    
    # Get the config file path from environment variable
    config_path = os.getenv("OAI_CONFIG_LIST")
    config_list = config_list_from_json(config_path)
    config_list = [config for config in config_list if config["model"] == model]
    print(config_list)
    optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)
    
    # Add debug printing
    def debug_step_hook(*args, **kwargs):
        summary = optimizer.summarize()
        instance = optimizer.problem_instance(summary)
        print("\n=== Problem Instance being sent to model ===")
        print(instance)
        print("\n=== End Problem Instance ===")
        return optimizer._step(*args, **kwargs)
    
    optimizer.step = debug_step_hook
    
    env = PongTracedEnv(env_name=env_name)

    rewards = []
    print("Optimization Starts")
    for i in range(n_optimization_steps):
        env.init()
        traj, error = rollout(env, horizon, policy)

        if error is None:
            feedback = f"Episode ends after {traj['steps']} steps with total score: {traj['total_reward']:.1f}"
            if traj['total_reward'] > 0:
                feedback += "\nGood job! You're scoring points against the opponent."
            elif traj['total_reward'] <= 0:
                feedback += "\nTry to improve paddle positioning to prevent opponent scoring."
            target = traj['observations'][-1]
            
            rewards.append(traj['total_reward'])
        else:
            feedback = error.exception_node.create_feedback()
            target = error.exception_node
        
        print(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {policy.parameter.data}")

        instruction = "In Pong, you control the right paddle and compete against the computer on the left. "
        instruction += "The goal is to keep deflecting the ball away from your goal and into your opponent's goal to maximize your score and win the game. "
        instruction += "You score one point when the opponent misses the ball or hits it out of bounds. "
        instruction += "The policy should move the right paddle up or down or NOOP to hit the ball. "
        # instruction += "The first player or team to score 21 points wins the game."
        
        optimizer.objective = instruction + optimizer.default_objective
        
        optimizer.zero_feedback()
        optimizer.backward(target, feedback)
        optimizer.step(verbose=verbose)
        
        print(f"Iteration: {i}, Feedback: {feedback}, Parameter: {policy.parameter.data}")
    
    print("Final Average Reward:", sum(rewards) / len(rewards))
    return rewards

rewards = optimize_policy(
    env_name="ALE/Pong-v5",
    horizon=1000,
    n_optimization_steps=10,
    memory_size=5,
    verbose='output',
    model="gpt-4o-mini"
)
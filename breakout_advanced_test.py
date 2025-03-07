import gymnasium as gym
import ale_py
import cv2
import numpy as np

# Global variable to store last known paddle x-position
last_paddle_x = None

def process_image(obs):
    """Process the grayscale image to detect the ball and paddle in Breakout.
    
    Args:
        obs: Grayscale image of the game screen
        
    Returns:
        dict: Dictionary containing position of ball and paddle found in the image
    """
    global last_paddle_x
    # Crop to the game area (removing top score area and side borders)
    cropped = obs[34:194, 15:147]
    
    # Use a lower threshold value to capture bright objects
    _, thresh = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_pos = None
    paddle_pos = None
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Detect the ball: a small, roughly square blob in the upper region
        if 1 < area < 10 and abs(w - h) < 3 and y < 100:
            ball_pos = [x + w // 2, y + h // 2, w, h]
        
        # Detect the paddle: a wide, horizontal object near the bottom
        if y > 110 and w > 15 and w > h * 3:
            paddle_pos = [x + w // 2, y + h // 2, w, h]
            last_paddle_x = paddle_pos[0]
    
    # If paddle detection fails, use the last known paddle x-position
    if paddle_pos is None and last_paddle_x is not None:
        paddle_pos = [last_paddle_x, 120, 16, 4]  # Use typical paddle dimensions
    
    return {"ball_pos": ball_pos, "paddle_pos": paddle_pos}


class BreakoutTestEnv:
    def __init__(self, env_name="ALE/Breakout-v5", render_mode="human", obs_type="grayscale"):
        self.env = gym.make(env_name, render_mode=render_mode, obs_type=obs_type)
        self.obs_type = obs_type
        
    def reset(self):
        obs, _ = self.env.reset()
        return process_image(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return process_image(obs), reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

def policy(obs):
    '''
    A policy that moves the paddle horizontally towards the predicted future position of the ball.
    It uses the difference in consecutive ball positions to estimate velocity and then predicts where the ball
    will be in a few steps. If the ball is not detected, the policy returns FIRE to launch the ball.

    Action mapping for Breakout:
    0 : NOOP
    1 : FIRE
    2 : Move right
    3 : Move left

    Args:
        obs (dict): A dictionary with keys "ball_pos" and "paddle_pos" representing the positions of the ball and paddle.
    Output:
        action (int): The action to take.
    '''
    ball_pos = obs["ball_pos"]
    paddle_pos = obs["paddle_pos"]

    # If the ball is not detected, attempt to launch it.
    if ball_pos is None:
        return 1  # FIRE

    # Use a function attribute to store the last observed ball position for velocity estimation.
    if not hasattr(policy, "last_ball_pos"):
        policy.last_ball_pos = ball_pos
        predicted_ball_x = ball_pos[0]
    else:
        # Calculate ball velocity in the x-direction based on the previous frame.
        vx = ball_pos[0] - policy.last_ball_pos[0]
        # Predict the ball's future x-position using a prediction horizon (tunable parameter).
        prediction_horizon = 5
        predicted_ball_x = ball_pos[0] + vx * prediction_horizon
        policy.last_ball_pos = ball_pos

    paddle_x = paddle_pos[0]

    # Use a tolerance threshold to avoid jittery moves.
    tolerance = 2
    if predicted_ball_x < paddle_x - tolerance:
        return 3  # Move left
    elif predicted_ball_x > paddle_x + tolerance:
        return 2  # Move right
    else:
        return 0  # NOOP

def test_policy(env, num_episodes=10, steps_per_episode=4000):
    rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action = policy(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
        rewards.append(episode_reward)
    
    print(f"\nAverage reward over {num_episodes} episodes: {sum(rewards) / num_episodes}")
    return rewards

if __name__ == "__main__":
    env = BreakoutTestEnv(render_mode='human')  # No rendering
    try:
        test_policy(env)
    finally:
        env.close()

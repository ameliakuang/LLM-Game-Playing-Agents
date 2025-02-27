import gymnasium as gym
import ale_py
import cv2
import numpy as np

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


class PongTestEnv:
    def __init__(self, env_name="ALE/Pong-v5", render_mode="human", obs_type="grayscale"):
        self.env = gym.make(env_name, render_mode=render_mode, obs_type=obs_type)
        self.obs_type = obs_type
        self.prev_ball_y = None
        
    def reset(self):
        obs, _ = self.env.reset()
        return process_image(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return process_image(obs), reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

def policy(obs):
    ball_pos = obs["ball_pos"]
    paddle_pos = obs["paddle_pos"]

    if ball_pos and paddle_pos:

        paddle_center = paddle_pos[1] + paddle_pos[3] / 2

        if paddle_center < ball_pos[1]:
            return 3  # move paddle up
        elif paddle_center > ball_pos[1]:
            return 2  # move paddle down
        else:
            return 0  # NOOP

    return 0  # NOOP

def test_policy(env, num_episodes=10, steps_per_episode=4000):
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
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    env = PongTestEnv(render_mode=None)  # No rendering
    try:
        test_policy(env)
    finally:
        env.close()

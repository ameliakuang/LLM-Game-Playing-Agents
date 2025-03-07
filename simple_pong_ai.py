import gymnasium as gym
import ale_py
import numpy as np
import cv2


def process_frame(gray):
    """ Extracts the ball and paddle positions from the grayscale frame. """
    # Crop relevant part of the frame (excluding scores and borders)
    gray = gray[34:194, 15:147]  # Crop game area
    
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
            ball_pos = (x, y)
        # Agent paddle is on the right side
        elif x > 100:  # Right paddle (agent)
            paddle_pos = (x, y)
    
    return ball_pos, paddle_pos

def simple_pong_ai(observation):
    """ Heuristic policy: Moves paddle towards the ball. """
    ball_pos, paddle_pos = process_frame(observation)
    
    action = 0  # Default action
    
    if ball_pos and paddle_pos:
        ball_y = ball_pos[1]
        paddle_y = paddle_pos[1]
        
        if paddle_y + 10 < ball_y:  # Paddle is below the ball, move up
            action = 3  
        elif paddle_y > ball_y + 10:  # Paddle is above the ball, move down
            action = 2
        
        # Print state information
        print(f"\rBall position: {ball_pos}, Paddle position: {paddle_pos}, Action: {action}", end="")
    else:
        print("\rNo detection - Ball or paddle not found", end="")
    
    return action

if __name__ == "__main__":

    gym.register_envs(ale_py)

    # Create the Pong environment
    env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="grayscale")

    # Run the game loop
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    episode_steps = 0
    total_reward = 0

    done = False
    while not done:
        action = simple_pong_ai(obs)  # Get action from heuristic AI
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"info: {info}, reward: {reward}")
        total_reward += reward
        episode_steps += 1
        
        if terminated or truncated:
            print(f"\nEpisode finished after {episode_steps} steps. Total reward: {total_reward}")
        
        done = terminated or truncated

    env.close()

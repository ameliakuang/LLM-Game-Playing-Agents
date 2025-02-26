import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from pong_env import PongEnv

def evaluate_model(model_path, algo_name, num_episodes=5):
    """Evaluate a trained model and display gameplay."""
    # Create environment with rendering
    env = PongEnv(render_mode="human")
    
    # Load the appropriate model
    if algo_name == "DQN":
        model = DQN.load(model_path)
    elif algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "A2C":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Run evaluation episodes
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
        rewards.append(episode_reward)
    
    env.close()
    
    # Print summary statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\n{algo_name} Summary:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Evaluate each algorithm
    algorithms = ["DQN", "PPO", "A2C"]
    results = {}
    
    for algo in algorithms:
        model_path = f"models/{algo}/{algo}_final.zip"
        if os.path.exists(model_path):
            print(f"\nEvaluating {algo}...")
            mean_reward, std_reward = evaluate_model(model_path, algo)
            results[algo] = (mean_reward, std_reward)
        else:
            print(f"\nNo trained model found for {algo}")
    
    # Print comparison
    print("\nComparison of all models:")
    for algo, (mean, std) in results.items():
        print(f"{algo}: {mean:.2f} +/- {std:.2f}")

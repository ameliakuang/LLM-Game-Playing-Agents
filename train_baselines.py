import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import wandb
from wandb.integration.sb3 import WandbCallback
from pong_env import PongEnv
import gymnasium as gym
import ale_py    

gym.register_envs(ale_py)

def make_env():
    """Create and wrap the Pong environment."""
    env = PongEnv(render_mode=None)  # No rendering during training
    env = Monitor(env)  # Adds episode stats
    return env

def train_model(algo_name, total_timesteps=1_000_000):
    """Train a specific algorithm."""
    # Create log directory
    model_dir = f"models/{algo_name}"
    log_dir = f"logs/{algo_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb
    run = wandb.init(
        project="cs224n_llm_agent",
        name=f"pong_{algo_name}",
        sync_tensorboard=True,
        config={
            "algorithm": algo_name,
            "total_timesteps": total_timesteps,
        },
        monitor_gym=True,
    )

    # Create environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for temporal information

    # Initialize the model based on algorithm
    if algo_name == "PPO":
        # Note: PPO batch_size should be a factor of n_steps * n_envs
        # Here we have n_steps=128 and n_envs=1, so batch_size=128
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=128 * env.num_envs,  # Adjusted to match n_steps * n_envs
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    elif algo_name == "A2C":
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            ent_coef=0.01,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix=algo_name,
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[WandbCallback(), checkpoint_callback],
        progress_bar=True,
    )

    # Save the final model
    model.save(f"{model_dir}/{algo_name}_final")

    # Evaluate the model
    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    wandb.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward
    })

    # Close environments
    env.close()
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    # Train each algorithm
    algorithms = ["PPO", "A2C"]
    for algo in algorithms:
        print(f"\nTraining {algo}...")
        train_model(algo)

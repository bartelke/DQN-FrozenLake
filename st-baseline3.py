import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

# Parametry DQN
params = {
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 32,
    "memory_size": 10000,
    "learning_rate": 0.001,
    "episodes": 1000
}

# Stwórz środowisko FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)

# Inicjalizacja modelu DQN
model = DQN(
    "MlpPolicy",
    env,
    gamma=params["gamma"],
    learning_rate=params["learning_rate"],
    buffer_size=params["memory_size"],
    batch_size=params["batch_size"],
    exploration_initial_eps=params["epsilon_start"],
    exploration_final_eps=params["epsilon_min"],
    exploration_fraction=(1.0 - params["epsilon_min"]) / params["episodes"],
    target_update_interval=500,
    train_freq=(1, "step"),
    gradient_steps=1,
    verbose=1,
    tensorboard_log="./dqn_frozenlake_tensorboard/"
)

# Trening modelu
time_steps = params["episodes"] * 50  # Przyjmujemy 100 kroków na epizod
print("Rozpoczynanie treningu...")
model.learn(total_timesteps=time_steps, log_interval=10)
print("Trening zakończony!")
import torch
import gymnasium as gym
import pickle
from dqn_agent import DQNAgent, train_agent

# Starting parameters
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

if __name__ == "__main__":
    # Create basic FrozenLake map
    env = gym.make("FrozenLake-v1", is_slippery=False)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Initialize agent obj
    agent = DQNAgent(state_size, action_size, params)

    # call the whole training process
    train_agent(env, agent, params)

    # save trained model
    torch.save(agent.main_network.state_dict(), "dqn_frozenlake.pth")
    print("Model został zapisany do pliku 'dqn_frozenlake.pth'.")

    # save the whole agent obj
    with open("dqn_agent.pkl", "wb") as file:
        pickle.dump(agent, file)
    print("Agent został zapisany do pliku 'dqn_agent.pkl'.")
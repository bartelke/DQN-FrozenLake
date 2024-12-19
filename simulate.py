import torch
import gymnasium as gym
import pygame
import pickle
import time
from dqn_agent import DQNAgent

def visualize_with_pygame(env, agent):
    # Init pygame and screen
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Frozen Lake Visualization")

    # start game
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # agent's actions
        action = agent.act(torch.eye(agent.state_size)[state].squeeze(0).numpy())
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        # render each game state
        env.render()
        time.sleep(0.5)

    print(f"Total Reward: {total_reward}")
    pygame.quit()


# load trained agent from files
def load_agent(filepath):
    with open(filepath, "rb") as file:
        agent = pickle.load(file)
    print(f"Agent został wczytany z pliku '{filepath}'.")
    return agent

if __name__ == "__main__":
    # create env
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # load agent
    agent = load_agent("dqn_agent.pkl")

    visualize_with_pygame(env, agent)

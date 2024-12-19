import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network class
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, params):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon_start"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.batch_size = params["batch_size"]
        self.memory_buffer = deque(maxlen=params["memory_size"])
        self.main_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=params["learning_rate"])
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.main_network(state)
        return torch.argmax(q_values, dim=1).item()
    
    def act_deterministic(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.main_network(state)
        return torch.argmax(q_values, dim=1).item()
    
    # save experiences in the memory buffer
    def store(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    # main network training logic
    def train(self, done):
        # If the memory does not have enough samples for a batch, return None
        if len(self.memory_buffer) < self.batch_size:
            return None 
        
        # get random samples for batch (from previous experience)
        batch = random.sample(self.memory_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert everything to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

         # Compute Q-values for the current states and actions using the main network
        q_values = self.main_network(states).gather(1, actions).squeeze(1)

        # Compute the maximum Q-value for the next states using the target network
        next_q_values = self.target_network(next_states).max(1)[0]

         # Calculate the target Q-values based on the Bellman equation
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()

        # typical networks thinks: compute gradiens and update weights
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

# whole training process
def train_agent(env, agent, params):
    mse_history = []
    reward_history = []
    steps_history = []
    total_rewards = [] 
    total_steps = 0
    csv_file = "training_data.csv"

    # CSV headers
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Steps", "Avg Steps (Last 100)", "Avg Reward (Last 100)"])

    for episode in range(params["episodes"]):
        state, _ = env.reset()
        state = torch.eye(agent.state_size)[state].squeeze(0).numpy()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.eye(agent.state_size)[next_state].squeeze(0).numpy()
            agent.store(state, action, reward, next_state, done)

            mse_loss = agent.train(done)
            if mse_loss is not None:
                mse_history.append(mse_loss)

            state = next_state
            total_reward += reward
            steps += 1

        # save some stats
        steps_history.append(steps)
        agent.update_target_network()
        total_rewards.append(total_reward)
        total_steps += steps

        # Every 10 episodes count average stats (last 100)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_steps = np.mean(steps_history[-100:])

            # write in CSV file
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, total_steps, avg_steps, avg_reward])

            print(f"Episode {episode + 1}/{params['episodes']}, Total Steps: {total_steps}, "
                  f"Avg Steps (last 100 episodes): {avg_steps:.2f}, Avg Reward (last 100 episodes): {avg_reward:.2f}")

    # Plot MSE after the end of the training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(mse_history) + 1), mse_history)
    plt.title("Historia wartości MSE")
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.show()

    return total_rewards

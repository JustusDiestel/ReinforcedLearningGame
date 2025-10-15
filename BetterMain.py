import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
import numpy as np
from collections import deque

# --- Environment ---
env = gym.make("CartPole-v1", render_mode="human")

# --- QNet ---
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
qnet = QNet(state_dim, action_dim)
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
criterion = nn.MSELoss()
replay_buffer = deque(maxlen=5000)

# --- Hyperparameter ---
gamma = 0.99
epsilon = 1.0
episodes = 200

# --- Training ---
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # epsilon-greedy: manchmal Zufall, manchmal beste Aktion
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = qnet(torch.tensor(state, dtype=torch.float32)).argmax().item()

        next_state, reward, done, trunc, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Training mit zufälligen Stichproben aus dem Replay Buffer
        if len(replay_buffer) > 64:
            batch = random.sample(replay_buffer, 64)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(s, dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.int64)

            q_values = qnet(s).gather(1, a.unsqueeze(1)).squeeze()
            target = r + gamma * qnet(ns).max(1)[0] * (1 - d)
            loss = criterion(q_values, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(0.05, epsilon * 0.995)  # weniger Zufall im Laufe des Trainings
    print(f"Ep {ep:03d} | Reward: {total_reward:5.1f} | Eps: {epsilon:.2f}")

env.close()
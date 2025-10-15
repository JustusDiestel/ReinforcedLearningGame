import gymnasium as gym
import time

import torch
import torch.nn as nn
import QNetwork as QNet


env = gym.make("CartPole-v1", render_mode="human")



for episode in range(3):
    state, _ = env.reset()
    total_reward = 0
    print("Startzustand:", state)
    for t in range(200):
        angle = state[2]
        action = 1 if angle > 0 else 0

        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        time.sleep(0.2)

        if done:
            print(f"Episode {episode+1} beendet mit Reward: {total_reward}")
            break

    env.close()

    state_dim = 4  # CartPole hat 4 Zustandswerte
    action_dim = 2  # links oder rechts
    qnet = QNet.QNet(state_dim, action_dim)
    sample_state = torch.tensor([0.1, 0.0, 0.05, 0.02])
    print("Q-Werte:", qnet(sample_state))
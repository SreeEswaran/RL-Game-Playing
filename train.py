import torch
import gym
from utils.agent import DQNAgent
from utils.environment import make_env
from config import EPISODES, LR, GAMMA, EPSILON_DECAY

def train():
    env = make_env()
    agent = DQNAgent(env.action_space.n)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=LR)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.optimize(optimizer, GAMMA)
            state = next_state
            if done:
                print(f"Episode {episode} finished")

if __name__ ==

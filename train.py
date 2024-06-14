import numpy as np
from environment.py import SimpleGame
from agent.py import QLearningAgent
import os

def train_agent(episodes=1000):
    env = SimpleGame()
    agent = QLearningAgent(state_size=11, action_size=2)
    log = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        log.append(total_reward)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    os.makedirs('results', exist_ok=True)
    np.save('results/model.npy', agent.q_table)
    with open('results/training_log.txt', 'w') as f:
        for reward in log:
            f.write(f"{reward}\n")

if __name__ == "__main__":
    train_agent()

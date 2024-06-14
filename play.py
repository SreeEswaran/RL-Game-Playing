import numpy as np
from environment.py import SimpleGame
from agent.py import QLearningAgent

def play_game():
    env = SimpleGame()
    q_table = np.load('results/model.npy')
    agent = QLearningAgent(state_size=11, action_size=2)
    agent.q_table = q_table
    
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
        
        if done:
            break

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    play_game()

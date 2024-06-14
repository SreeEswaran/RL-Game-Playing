import numpy as np
class SimpleGame:
    def __init__(self):
        self.state = 0
        self.end_state = 10
        self.reset()
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 1:
            self.state += 1
        elif action == 0:
            self.state -= 1
        
        reward = -1
        done = False
        
        if self.state == self.end_state:
            reward = 10
            done = True
        elif self.state < 0:
            self.state = 0
            
        return self.state, reward, done

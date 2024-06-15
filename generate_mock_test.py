import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Generate and save a mock Q-table
q_table = np.random.rand(11, 2)
np.save('results/model.npy', q_table)

# Generate and save a mock training log
training_log = [-100, -95, -89, -76, -65, -50, -40, -25, -15, -5]
with open('results/training_log.txt', 'w') as f:
    for reward in training_log:
        f.write(f"{reward}\n")

print("Mock results generated in the 'results' folder.")

import gym
import pandas as pd
import numpy as np
from gym import spaces
import matplotlib as plt
import torch


# Demonstration
env = gym.envs.make("CartPole-v1")
def get_screen():
    ''' Extract one step of the simulation.'''
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    return torch.from_numpy(screen)
# Speify the number of simulation steps
num_steps = 2
# Show several steps
for i in range(num_steps):
    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('CartPole-v0 Environment')
    plt.xticks([])
    plt.yticks([])
    plt.show()
import torch
import gym

class Config(object):
    """docstring for Config"""
    def __init__(self):
        self.HIDDEN_LAYER = 128
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99 # discount factor
        self.EPS_START = 1.0 # init exploration rate
        self.EPS_END = 0.1 # final exploration
        self.EPS_DECAY = 200 # decay
        self.TARGET_UPDATE = 100 # target network update freq
        self.num_episodes = 500
        self.LR = 0.001 # adam lr
        self.initial_exploration = 1000
        self.env = gym.make('CartPole-v0').unwrapped
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 500
        self.memory_cap = 1000
        self.END_EPSISODE = 300
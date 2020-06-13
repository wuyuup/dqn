import torch
from config import Config
from agent import DQNAgent, DoubleDQNAgent


args = Config()
torch.manual_seed(args.seed)
agent = DQNAgent(args)


# training

agent.train()

print('complete!')
import torch
from config import Config
from agent import DQNAgent, DoubleDQNAgent

def main():
    args = Config()
    torch.manual_seed(args.seed)
    agent = DoubleDQNAgent(args)
    # training
    agent.train()
    print('complete!')

if __name__ == '__main__':
    main()
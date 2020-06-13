import random
import torch.nn.functional as F
import torch
import torch.optim as optim
from model import DQN, ReplayMemory
from itertools import count
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DQNAgent(object):
    """docstring for DQNAgent"""
    def __init__(self, args):
        self.args = args
        self.env = self.args.env
        self.env.seed(self.args.seed)



    def __select_action(self):
        p = random.random()
        if p > self.eps:
            return self.q_net(self.state).max(1)[1].numpy()[0]
        else:
            return self.env.action_space.sample()

    def __optimize_model(self):
        # no need to optimize 
        if len(self.memory) < self.args.BATCH_SIZE:
            return
        # extract batch
        batch = self.memory.sample(self.args.BATCH_SIZE)
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.Tensor(batch.reward)
        mask_batch = torch.Tensor(batch.mask)
        # gather function seleces the output corresponding to action_batch
        q_values = self.q_net(state_batch).squeeze(1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # q val from target
        next_state_values = self.target_net(next_state_batch).squeeze(1).max(1)[0]
        # compute expected Q vals
        target_q_values = mask_batch * (next_state_values * self.args.GAMMA) + reward_batch
        # loss
        loss = F.mse_loss(q_values, target_q_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def __step(self, t):
        self.steps_done += 1
        action = self.__select_action()
        next_state, reward, done, _ = self.env.step(action)
        next_state = torch.Tensor(next_state).unsqueeze(0)
        if done and t < self.args.END_EPSISODE:
            reward = -1
        mask = 0 if done else 1
        self.memory.push(self.state, next_state, action, reward, mask)
        self.state = next_state
        if self.steps_done > self.args.initial_exploration:
            self.eps -= 0.00005
            self.eps = max(self.eps, self.args.EPS_END)
            self.__optimize_model()
            if self.steps_done % self.args.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return done


    def train(self):
        # feature dimension
        state_dim = self.env.observation_space.shape[0]
        # number of actions
        n_actions = self.env.action_space.n

        self.q_net = DQN(state_dim, n_actions).to(self.args.device)
        self.target_net = DQN(state_dim, n_actions).to(self.args.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.train()
        self.q_net.train()
        # self.optimizer = optim.RMSprop(self.q_net.parameters())
        self.optimizer = optim.Adam(self.q_net.parameters(), self.args.LR)
        self.memory = ReplayMemory(self.args.memory_cap)
        self.steps_done = 0
        self.eps = self.args.EPS_START
        self.episode_durations = []

        for i_episode in range(self.args.num_episodes):
            self.state = self.env.reset()
            self.state = torch.Tensor(self.state).to(self.args.device).unsqueeze(0)
            for t in count():
                done = self.__step(t)
                if done or t >= self.args.END_EPSISODE:
                    self.episode_durations.append(t+1)
                    if len(self.episode_durations) > 100:
                        ave = np.mean(self.episode_durations[-100:])
                    else:
                        ave = np.mean(self.episode_durations)
                    if i_episode % 10 == 0:
                        print("[Episode {:>5}]  steps: {:>5}   ave: {:>5}".format(i_episode, t, ave))
                    break

        plt.figure()
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype = torch.float)
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        plt.plot(means.numpy())
        plt.title('training')
        plt.xlabel('episode')
        plt.ylabel('duration')
        plt.savefig('res.png')



class DoubleDQNAgent(DQNAgent):
    """docstring for DoubleDQNAgent"""
    def __init__(self, args):
        super(DoubleDQNAgent, self).__init__(args)
    
    def __optimize_model(self):
        # no need to optimize 
        if len(self.memory) < self.args.BATCH_SIZE:
            return
        # extract batch
        batch = self.memory.sample(self.args.BATCH_SIZE)
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.Tensor(batch.reward)
        mask_batch = torch.Tensor(batch.mask)
        # gather function seleces the output corresponding to action_batch
        q_values = self.q_net(state_batch).squeeze(1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # double DQN 
        next_action_batch = self.q_net(next_state_batch).squeeze(1).max(1)[1]
        output_next_state_batch = self.target_net(next_state_batch).squeeze(1)
        next_state_values_double = output_next_state_batch.gather(1, next_action_batch.unsqueeze(1)).squeeze(1)
        # compute expected Q vals
        target_q_values = mask_batch * (next_state_values_double * self.args.GAMMA) + reward_batch
        loss = F.mse_loss(q_values, target_q_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

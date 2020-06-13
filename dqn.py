import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# HIDDEN_LAYER = 164
BATCH_SIZE = 32
GAMMA = 0.99 # discount factor
EPS_START = 1.0 # init exploration rate
EPS_END = 0.1 # final exploration
EPS_DECAY = 200 # decay
TARGET_UPDATE = 10000 # target network update freq
num_episodes = 100000
LR = 0.001 # adam lr


env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#  replay buffer

Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))

class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity # data stored in a circle

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




'''
# network: extract from screen
class DQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, h, w, outputs_feature):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2) # input: 3
        self.bn1 = nn.BatchNorm2d(16) # number of features: 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size-1)-1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs_feature)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1)) # view as a vector
'''


# nn: take state from env as input
class DQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, input_feature, outputs_feature):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(input_feature, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, outputs_feature)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)



# training

# feature dimension
state_dim = len(env.reset())

# number of actions
n_actions = env.action_space.n


q_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval() # Sets the module in evaluation mode

optimizer = optim.RMSprop(q_net.parameters())
# optimizer = optim.Adam(q_net.parameters(), LR)

memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    prob = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
    steps_done += 1
    if sample > prob:
        with torch.no_grad():
            # __call__
            # max(1) gives largest column value with index in the second col
            # return q_net(torch.autograd.Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1,1)
            return q_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype = torch.float)
    plt.title('training')
    plt.xlabel('episode')
    plt.ylabel('duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(1e-8)


def optimize_model():
    # no need to optimize 
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # check the unpack!
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # gather function seleces the output corresponding to action_batch
    state_action_values = q_net(state_batch).gather(1, action_batch)
    # q val from target
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    # compute expected Q vals
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize
    optimizer.zero_grad()
    loss.backward()
    # for param in q_net.parameters():
    #     param.grad.data.clamp_(-1, 1) # Clamp all elements in input into the range

    optimizer.step()



num_param_update = 0

for i_episode in range(num_episodes):
    env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    state = env.reset()
    for t in count():
        action = select_action(torch.FloatTensor([state]))
        next_state, reward, done, _ = env.step(action.item())

        if done:
            if t < 30:
                reward -= 10
            else:
                reward = -1
        if t > 100:
            reward += 1
        if t > 200:
            reward += 1
        if t > 300:
            reward += 1

        reward = max(-1.0, min(reward, 1.0))

        reward = torch.FloatTensor([reward], device=device)

        # store in buffer
        memory.push(torch.FloatTensor([state]), action, torch.FloatTensor([next_state]), reward)

        state = next_state

        optimize_model()

        num_param_update += 1
        # print(num_param_update)

        if done or t >= 300:
            episode_durations.append(t+1)
            # plot_durations()
            if len(episode_durations) > 100:
                ave = np.mean(episode_durations[-100:])
            else:
                ave = np.mean(episode_durations)
            print("[Episode {:>5}]  steps: {:>5}   ave: {:>5}".format(i_episode, t, ave))
            break

        if num_param_update % TARGET_UPDATE == 0:
            # print("#################### update target ####################")
            target_net.load_state_dict(q_net.state_dict())


print('complete')

#env.render()
#env.close()
#plt.ioff()
#plt.show()


# Agent
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from collections import deque
from model import DQNModel
import config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent():
    def __init__(self):
        self.model = DQNModel(config.state_size[-1]*config.stack_frame, "main_model", config.action_size).to(device)
        self.target_model = DQNModel(config.state_size[-1], "target_model", config.action_size).to(device)

        self.memory = deque(maxlen=config.mem_maxlen)
        self.obs_set = deque(maxlen=config.skip_frame*config.stack_frame)

        self.epsilon = config.epsilon_init
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.update_target()

        if config.load_model == True:
            self.model.load_state_dict(torch.load(config.load_path))

    # action selection based on epsilon greedy method
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # random action
            return np.random.randint(config.action_size)

        else:
            # action from network
            model_out = self.model(torch.from_numpy(state))
            predict = torch.argmax(model_out, dim=1)
            return np.asscalar(predict)

    # skip and stack frame
    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)
        state = np.zeros([config.state_size[0], config.state_size[1], config.state_size[2]*stack_frame])

        # obs stacking every skip frame
        for i in range(config.stack_frame):
            state[:, :, config.state_size[2]*i : config.state_size[2]*(i+1)] = self.obs_set[-1 - (config.skip_frame*i)]

        return np.uint8(state)

    # data augmentation to replay memory (state, action, reward, next_state, done)
    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append([self, state, action, reeard, next_state, done])


    # save network model
    def save_model(self):
        torch.save(self.model.state_dict(), config.save_path+'/model.pth')
        print("Save Model: {}".format(config.save_path))


    # train network model
    def train_model(self, done):
        # sample minibatch for training
        mini_batch = random.sample(self.memory, config.batch_size)

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for i in range(config.batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            next_states.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])



        # calculate target value for training
        Q_ = self.model(torch.FloatTensor(states).to(device))
        Q = Q_.cpu().detach().numpy()
        target_Q = self.target_model(torch.FloatTensor(next_states).to(device)).cpu().detach().numpy()

        for i in range(config.batch_size):
            if dones[i]:
                Q[i][actions[i]] = rewards[i]
            else:
                Q[i][actions[i]] = rewards[i] + config.discount_factor * np.amax(target_Q[i])

        # compute loss
        loss = F.smooth_l1_loss(target_Q, Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # update target network
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

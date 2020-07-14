# Agent
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import config

from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter

class DQNAgent():
    def __init__(self, model, model_target, optimizer, device):
        self.model = model
        self.model_target = model_target

        self.memory = deque(maxlen=config.mem_maxlen)
        self.obs_set = deque(maxlen=config.skip_frame*config.stack_frame)

        self.epsilon = config.epsilon_init
        self.optimizer = optimizer
        self.device = device

        self.writer = SummaryWriter('{}'.format(config.save_path))

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
            # print(np.shape(state)) # (3, 80, 80)
            # print(torch.from_numpy(state).unsqueeze(0).size()) # (1, 3, 80, 80)
            Q = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device))
            return np.argmax(Q.cpu().detach().numpy())

    # skip and stack frame
    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)
        state = np.zeros([config.state_size[2]*config.stack_frame, config.state_size[0], config.state_size[1]])

        # obs stacking every skip frame
        for i in range(config.stack_frame):
            state[config.state_size[2]*i : config.state_size[2]*(i+1), :, :] = self.obs_set[-1 - (config.skip_frame*i)]

        return np.uint8(state)

    # data augmentation to replay memory (state, action, reward, next_state, done)
    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # save network model
    def save_model(self):
        if not os.path.isdir(config.save_path):
            os.mkdir(config.save_path)
        torch.save(self.model.state_dict(), config.save_path+'/model.pth')
        print("Save Model: {}".format(config.save_path))


    # train network model
    def train_model_DQN(self, done):
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
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # calculate target value for training
        Q_pred = self.model(torch.FloatTensor(states).to(self.device)) # (batch_size, action_size) : (32, 4)
        Q_target = Q_pred.cpu().detach().numpy()
        Q_next_target = self.model_target(torch.FloatTensor(next_states).to(self.device)).cpu().detach().numpy()

        for i in range(config.batch_size):
            if dones[i]:
                Q_target[i][actions[i]] = rewards[i]
            else:
                Q_target[i][actions[i]] = rewards[i] + config.discount_factor * np.amax(Q_next_target[i])

        # compute loss
        loss = F.smooth_l1_loss(input=Q_pred, target=torch.from_numpy(Q_target).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # train network model
    def train_model_doubleDQN(self, done):
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
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # calculate target value for training
        Q_pred = self.model(torch.FloatTensor(states).to(self.device)) # (batch_size, action_size) : (32, 4)
        Q_target = Q_pred.cpu().detach().numpy()
        Q_next_target = self.model_target(torch.FloatTensor(next_states).to(self.device)).cpu().detach().numpy()

        Q_ = self.model(torch.FloatTensor(next_states).to(self.device)).cpu().detach().numpy()

        for i in range(config.batch_size):
            if dones[i]:
                Q_target[i][actions[i]] = rewards[i]
            else:
                action_ind = np.argmax(Q_[i])
                Q_target[i][actions[i]] = rewards[i] + config.discount_factor * Q_next_target[i][action_ind]

        # compute loss
        loss = F.smooth_l1_loss(input=Q_pred, target=torch.from_numpy(Q_target).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    #  target network
    def update_target(self):
        self.model_target.load_state_dict(self.model.state_dict())


    def writer_tb(self, loss, reward, episode):
        self.writer.add_scalar('Mean Loss', loss, episode)
        self.writer.add_scalar('Mean Reward', reward, episode)

import model
import config

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import numpy as np
import random
from collections import deque
import os

device = config.device

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의
class DQNAgent():
    def __init__(self, model, target_model, optimizer, device, algorithm):
        # 클래스의 함수들을 위한 값 설정
        if algorithm == ("_RND" or "_ICM"):
            self.model = model[0]
            self.model_a = model[1]
        else:
            self.model = model

        self.target_model = target_model
        self.optimizer = optimizer

        self.device = device
        self.algorithm = algorithm

        self.memory = deque(maxlen=config.mem_maxlen)
        self.obs_set = deque(maxlen=config.skip_frame*config.stack_frame)

        self.epsilon = config.epsilon_init

        if not config.load_model and config.train_mode:
            self.writer = SummaryWriter('{}'.format(config.save_path + self.algorithm))
        elif config.load_model and config.train_mode:
            self.writer = SummaryWriter('{}'.format(config.load_path))

        self.update_target()

        if config.load_model == True:
            try:
                self.model_a
            except:
                self.model.load_state_dict(torch.load(config.load_path+'/model.pth'))
            else:
                checkpoint = torch.load(config.load_path+'/model.pth')
                self.model.load_state_dict(checkpoint['model'])
                self.model_a.load_state_dict(checkpoint['model_a'])

            print("Model is loaded from {}".format(config.load_path+'/model.pth'))

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            return np.random.randint(0, config.action_size)
        else:
            with torch.no_grad():
            # 네트워크 연산에 따라 행동 결정
                Q = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device))
                return np.argmax(Q.cpu().detach().numpy())

    # 프레임을 skip하면서 설정에 맞게 stack
    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)

        state = np.zeros([config.state_size[2]*config.stack_frame, config.state_size[0], config.state_size[1]])

        # skip frame마다 한번씩 obs를 stacking
        for i in range(config.stack_frame):
            state[config.state_size[2]*i : config.state_size[2]*(i+1), :,:] = self.obs_set[-1 - (config.skip_frame*i)]

        return np.uint8(state)

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 네트워크 모델 저장
    def save_model(self, load_model, train_mode):
        if not load_model and train_mode: # first training
            os.makedirs(config.save_path + self.algorithm, exist_ok=True)
            try:
                self.model_a
            except:
                torch.save(self.model.state_dict(), config.save_path + self.algorithm +'/model.pth')
            else:
                torch.save({
                    'model': self.model.state_dict(),
                    'model_a': self.model_a.state_dict()
                }, config.save_path + self.algorithm +'/model.pth')

            print("Save Model: {}".format(config.save_path + self.algorithm))

        elif load_model and train_mode: # additional training
            try:
                self.model_a
            except:
                torch.save(self.model.state_dict(), config.load_path +'/model.pth')
            else:
                torch.save({
                    'model': self.model.state_dict(),
                    'model_a': self.model_a.state_dict()
                }, config.load_path +'/model.pth')

            print("Save Model: {}".format(config.load_path))

    # 학습 수행
    def train_model(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)

        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)

        # 타겟값 계산
        Q = self.model(state_batch)
        action_batch_onehot = torch.eye(config.action_size)[action_batch.type(torch.long)].cuda()
        acted_Q = torch.sum(Q * action_batch_onehot, axis=-1).unsqueeze(1)

        with torch.no_grad():
            target_next_Q = self.target_model(next_state_batch)
            max_next_Q = torch.max(target_next_Q, dim=1, keepdim=True).values
            target_Q = (1. - done_batch).view(config.batch_size, -1) * config.discount_factor * max_next_Q + reward_batch.view(config.batch_size, -1)

        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()

        loss = F.smooth_l1_loss(acted_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def write_scalar(self, loss, reward, maxQ, episode):
        self.writer.add_scalar('Mean_Loss', loss, episode)
        self.writer.add_scalar('Mean_Reward', reward, episode)
        self.writer.add_scalar('Max_Q', maxQ, episode)

    def write_scalar_ICM(self, loss, reward, maxQ, r_i, episode, loss_rl, loss_fm, loss_im):
        self.writer.add_scalar('Mean_Loss', loss, episode)
        self.writer.add_scalar('Mean_Reward', reward, episode)
        self.writer.add_scalar('Max_Q', maxQ, episode)
        self.writer.add_scalar('intrinsic_Reward', r_i, episode)
        self.writer.add_scalar('Mean_Loss_Rl', loss_rl, episode)
        self.writer.add_scalar('Mean_Loss_Fm', loss_fm, episode)
        self.writer.add_scalar('Mean_Loss_Im', loss_im, episode)

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action_noisy(self, state, step, train_mode):
        if step < config.start_train_step and train_mode:
            # 랜덤하게 행동 결정
            return np.random.randint(0, config.action_size)
        else:
            # 네트워크 연산에 따라 행동 결정
            Q = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device), torch.tensor(train_mode).to(self.device))
            return np.argmax(Q.cpu().detach().numpy())

    # 학습 수행
    def train_model_double(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for i in range(config.batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])

        # 타겟값 계산
        predict_Q = self.model(torch.FloatTensor(state_batch).to(self.device))
        target_Q = predict_Q.cpu().detach().numpy()
        target_nextQ = self.target_model(torch.FloatTensor(next_state_batch).to(self.device)).cpu().detach().numpy()

        Q_a = self.model(torch.FloatTensor(next_state_batch).to(self.device)).cpu().detach().numpy()
        max_Q = np.max(target_Q)

        with torch.no_grad():
            for i in range(config.batch_size):
                if done_batch[i]:
                    target_Q[i, action_batch[i]] = reward_batch[i]
                else:
                    action_ind = np.argmax(Q_a[i])
                    target_Q[i, action_batch[i]] = reward_batch[i] + config.discount_factor * target_nextQ[i][action_ind]

        loss = F.smooth_l1_loss(predict_Q.to(self.device), torch.from_numpy(target_Q).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q

    # 학습 수행
    def train_model_ICM(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)
        
        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)

        # ICM
        x_next_encode, x_fm, x_im = self.model_a(state_batch, next_state_batch, action_batch)

        # calculate intrinsic reward
        reward_i = (config.eta * 0.5) * torch.sum(torch.square(x_fm - x_next_encode), dim=1)

        Q = self.model(state_batch)
        action_batch_onehot = torch.eye(config.action_size)[action_batch.type(torch.long)].cuda()
        acted_Q = torch.sum(Q * action_batch_onehot, axis=-1).unsqueeze(1)

        with torch.no_grad():
            for i in range(config.batch_size):
                reward_batch[i] = (config.extrinsic_coeff * reward_batch[i]) + (config.intrinsic_coeff * reward_i[i])

            target_next_Q = self.target_model(next_state_batch)
            max_next_Q = torch.max(target_next_Q, dim=1, keepdim=True).values
            target_Q = (1. - done_batch).view(config.batch_size, -1) * config.discount_factor * max_next_Q + reward_batch.view(config.batch_size, -1)

        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()

        loss_rl = F.smooth_l1_loss(acted_Q, target_Q)
        # ICM related losses
        loss_fm = F.mse_loss(input=x_fm.to(self.device), target=x_next_encode.to(self.device))
        loss_im = F.cross_entropy(input=x_im.to(self.device), target=action_batch.to(device=self.device, dtype=torch.int64))

        loss = (config.lamb * loss_rl) + (config.beta * loss_fm) + ((1-config.beta) * loss_im)

        self.optimizer.zero_grad()
        loss_rl.backward(retain_graph=True)
        loss_fm.backward(retain_graph=True)
        loss_im.backward(retain_graph=True)

        self.optimizer.step()

        return loss.item(), max_Q, config.intrinsic_coeff*reward_i.cpu().detach().numpy(), loss_rl.item(), loss_fm.item(), loss_im.item()


    # 학습 수행
    def train_model_RND(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)

        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)

        # RND
        x_next_encode, x_next_encode_t = self.model_a(next_state_batch)

        # calculate intrinsic reward
        reward_i = (config.eta * 0.5) * torch.sum(torch.square(x_next_encode - x_next_encode_t), dim=1)

        Q = self.model(state_batch)
        # print(f"Q: {Q}")
        action_batch_onehot = torch.eye(config.action_size)[action_batch.type(torch.long)].cuda()
        acted_Q = torch.sum(Q * action_batch_onehot, axis=-1).unsqueeze(1)

        with torch.no_grad():
            for i in range(config.batch_size):
                reward_batch[i] = (config.extrinsic_coeff * reward_batch[i]) + (config.intrinsic_coeff * reward_i[i])

            target_next_Q = self.target_model(next_state_batch)
            max_next_Q = torch.max(target_next_Q, dim=1, keepdim=True).values
            target_Q = (1. - done_batch).view(config.batch_size, -1) * config.discount_factor * max_next_Q + reward_batch.view(config.batch_size, -1)

        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()

        loss_rl = F.smooth_l1_loss(acted_Q, target_Q)

        # RND loss
        loss_fm = F.mse_loss(input=x_next_encode.to(self.device), target=x_next_encode_t.to(self.device))

        loss = (config.lamb * loss_rl) + (config.beta * loss_fm)

        self.optimizer.zero_grad()
        loss_rl.backward(retain_graph=True)
        loss_fm.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item(), max_Q, config.intrinsic_coeff*reward_i.cpu().detach().numpy(), loss_rl.item(), loss_fm.item()

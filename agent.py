import model
import config

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from collections import deque
import os

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의
class DQNAgent():
    def __init__(self, model, target_model, optimizer, device, algorithm):
        # 클래스의 함수들을 위한 값 설정
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer

        self.device = device
        self.algorithm = algorithm

        self.memory = deque(maxlen=config.mem_maxlen)
        self.obs_set = deque(maxlen=config.skip_frame*config.stack_frame)

        self.epsilon = config.epsilon_init

        if not config.load_model:
            self.writer = SummaryWriter('{}'.format(config.save_path + self.algorithm))

        self.update_target()

        if config.load_model == True:
            self.model.load_state_dict(torch.load(config.load_path))
            print("Model is loaded from {}".format(config.load_path))

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
    def save_model(self, load_model):
        if not load_model:
            os.makedirs(config.save_path + self.algorithm, exist_ok=True)
            torch.save(self.model.state_dict(), config.save_path + self.algorithm +'/model.pth')
            print("Save Model: {}".format(config.save_path + self.algorithm))

    # 학습 수행
    def train_model(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)

        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)

        # print(f"\n [-] memory")
        # print(f"state_batch : {state_batch.shape}, {state_batch.dtype}")
        # print(f"action_batch : {action_batch.shape}, {action_batch.dtype}")
        # print(f"reward_batch : {reward_batch.shape}, {reward_batch.dtype}")
        # print(f"next_state_batch : {next_state_batch.shape}, {next_state_batch.dtype}")
        # print(f"done_batch : {done_batch.shape}, {done_batch.dtype}")

        # print(f"state_batch: {state_batch[0].shape}, next_state_batch: {next_state_batch[0].shape}")
        # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        #
        # for i, state in enumerate(state_batch[0]):
        #     axes[0, i].imshow(state)
        # for i, state in enumerate(next_state_batch[0]):
        #     axes[1, i].imshow(state)
        # plt.show()


        # 타겟값 계산
        Q = self.model(state_batch)
        # print(f"Q: {Q}")
        action_batch_onehot = torch.eye(3)[action_batch.type(torch.long)].cuda()
        acted_Q = torch.sum(Q * action_batch_onehot, axis=-1).unsqueeze(1)
        # print("acted_Q")
        # print(acted_Q)

        with torch.no_grad():
            target_next_Q = self.target_model(next_state_batch)
            max_next_Q = torch.max(target_next_Q, dim=1, keepdim=True).values
            target_Q = (1. - done_batch).view(config.batch_size, -1) * config.discount_factor * max_next_Q + reward_batch.view(config.batch_size, -1)

        # print(f"target_next_Q: {target_next_Q}")
        # print(f"max_next_Q: {max_next_Q}")
        # print("target_Q")
        # print(target_Q)

        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()
        # print(f"max_Q: {max_Q}")

        loss = F.smooth_l1_loss(acted_Q, target_Q)
        # print(f"loss: {loss}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("--------------------------------------------")

        return loss.item(), max_Q



    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def write_scalar(self, loss, reward, maxQ, episode):
        self.writer.add_scalar('Mean_Loss', loss, episode)
        self.writer.add_scalar('Mean_Reward', reward, episode)
        self.writer.add_scalar('Max_Q', maxQ, episode)

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
    def train_model_noisy(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)
        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)

        # 타겟값 계산
        Q = self.model(state_batch, train=True)
        action_batch_onehot = torch.eye(3)[action_batch.type(torch.long)].cuda()
        acted_Q = torch.sum(Q * action_batch_onehot, axis=-1).unsqueeze(1)

        with torch.no_grad():
            target_next_Q = self.target_model(next_state_batch, train=False)
            max_next_Q = torch.max(target_next_Q, dim=1, keepdim=True).values
            target_Q = (1. - done_batch).view(config.batch_size, -1) * config.discount_factor * max_next_Q + reward_batch.view(config.batch_size, -1)

        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()
        # print(f"max_Q: {max_Q}")

        loss = F.smooth_l1_loss(acted_Q, target_Q)
        # print(f"loss: {loss}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item(), max_Q

    def train_C51(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, config.batch_size)
        state_batch = torch.cat([torch.tensor([mini_batch[i][0]]) for i in range(config.batch_size)]).float().to(self.device)
        action_batch = torch.cat([torch.tensor([mini_batch[i][1]]) for i in range(config.batch_size)]).float().to(self.device)
        reward_batch = torch.cat([torch.tensor([mini_batch[i][2]]) for i in range(config.batch_size)]).float().to(self.device)
        next_state_batch = torch.cat([torch.tensor([mini_batch[i][3]]) for i in range(config.batch_size)]).float().to(self.device)
        done_batch = torch.cat([torch.tensor([mini_batch[i][4]]) for i in range(config.batch_size)]).float().to(self.device)


        z = torch.linspace(config.vmin, config.vmax, config.atoms).view(1, 1, config.atoms).to(self.device)

        prob = self.model(state_batch)
        action = action_batch.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, -1, config.atoms).long()
        prob_current_action = prob.gather(1, action).squeeze()
        dist = prob * z

        Q = torch.sum(dist, dim=-1)

        # target distribution 계산하기
        with torch.no_grad():
            target_dist = torch.zeros(config.batch_size, config.atoms, requires_grad=False).to(self.device)

            prob_next = self.target_model(next_state_batch)
            dist_next = prob_next * z
            Q_next = torch.sum(dist_next, dim=-1)

            # projection
            for i in range(config.batch_size):
                action_max = torch.argmax(Q_next[i])
                done = done_batch[i]

                r = reward_batch[i]

                if done:
                    Tz = r
                    # Bounding Tz
                    if Tz >= config.vmax:
                        Tz = config.vmax
                    elif Tz <= config.vmin:
                        Tz = config.vmin

                    b = (Tz - config.vmin) / self.model.delta_z
                    l = torch.floor(b).long()
                    u = torch.ceil(b).long()

                    target_dist[i, l] += (u - b)
                    target_dist[i, u] += (b - l)

                    if l==u:
                        target_dist[i, l] = 1

                else:
                    for j in range(config.atoms):
                        Tz = r + config.discount_factor * z[0, 0, j]
                        Tz = torch.clamp(Tz, config.vmin, config.vmax)
                        b = (Tz - config.vmin) / self.model.delta_z
                        l = torch.floor(b).long()
                        u = torch.ceil(b).long()

                        target_dist[i, l] += prob_next[i, action_max, j] * (u-b)
                        target_dist[i, u] += prob_next[i, action_max, j] * (b-l)

                    sum_target_dist = torch.sum(target_dist[i,:])
                    for j in range(config.atoms):
                        target_dist[i, j] = target_dist[i, j] / sum_target_dist

        target_Q = torch.sum(target_dist, dim=-1)
        max_Q = torch.mean(torch.max(target_Q, axis=0).values).cpu().numpy()

        loss = -(target_dist * prob_current_action.log()).sum(-1)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, max_Q

import config

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import numpy as np 
import random
from collections import deque

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent():
    def __init__(self, model, target_model, optimizer, device):
        # 클래스의 함수들을 위한 값 설정 
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer

        self.device = device

        self.memory = deque(maxlen=config.mem_maxlen)
        self.obs_set = deque(maxlen=config.skip_frame*config.stack_frame)
   
        self.epsilon = config.epsilon_init

        self.writer = SummaryWriter('{}'.format(config.save_path))

        self.update_target()

        if config.load_model == True:
            self.model.state_dict(torch.load(config.load_path))
            print("Model is loaded from {}".format(config.load_path))

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            return np.random.randint(0, config.action_size)
        else:
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
    def save_model(self):
        torch.save(self.model.state_dict(), config.save_path+'/model.pth')
        print("Save Model: {}".format(config.save_path))

    # 학습 수행 
    def train_model(self, done):
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

        for i in range(config.batch_size):
            if done_batch[i]:
                target_Q[i, action_batch[i]] = reward_batch[i]
            else:
                target_Q[i, action_batch[i]] = reward_batch[i] + config.discount_factor * np.amax(target_nextQ[i])

        loss = F.smooth_l1_loss(predict_Q.to(self.device), torch.from_numpy(target_Q).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 타겟 네트워크 업데이트 
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def write_scalar(self, loss, reward, episode):
        self.writer.add_scalar('Mean Loss', loss, episode)
        self.writer.add_scalar('Mean Reward', reward, episode)
        self.writer.flush()
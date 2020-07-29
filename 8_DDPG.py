# Import Libraries
import numpy as np
import random
import time
import datetime
import os
from collections import deque
from mlagents.envs import UnityEnvironment

import torch
import torch.optim as optim

import agent
import model
import config

# Main function
if __name__ == '__main__':
    # set unity environment path (file_name)
    env = UnityEnvironment(file_name=config.env_name)
    # env = UnityEnvironment(file_name=config.env_name, worker_id=np.random.randint(100000))

    # setting brain for unity
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    train_mode = config.train_mode

    device = config.device

    actor = model.Actor(config.action_size, "main").to(device)
    target_actor = model.Actor(config.action_size, "target").to(device)
    critic = model.Critic(config.action_size, "main").to(device)
    target_critic = model.Critic(config.action_size, "target").to(device)

    optimizer_actor = optim.Adam(actor.parameters(), lr=config.actor_lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=config.critic_lr)

    algorithm = "_DDPG"
    agent = agent.DDPGAgent(actor, critic, target_actor, target_critic, optimizer_actor, optimizer_critic, device, algorithm)

    # Initialize target networks
    agent.hard_update_target()

    step = 0
    episode = 0
    reward_list = []
    loss_critic_list = []
    loss_actor_list = []
    max_Q_list = []

    # Reset Unity environment and set the train mode according to the environment setting (env_config)
    env_info = env.reset(train_mode=train_mode, config=config.env_config)[default_brain]

    # Game loop
    while step < config.run_step + config.test_step:
        # Initialize state, episode_rewards, done
        state = env_info.vector_observations[0]
        episode_rewards = 0
        done = False

        # loop for an episode
        while not done:
            if step == config.run_step:
                train_mode = False
                env_info = env.reset(train_mode=train_mode)[default_brain]

            # Decide action and apply the action to the Unity environment
            action = agent.get_action(state.astype(np.float32), train_mode)
            env_info = env.step(action)[default_brain]

            # Get next state, reward, done information
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            # Save data in replay memory while train mode
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.0)

            # Update state information
            state = next_state
            step += 1

            if step > config.start_train_step and train_mode:
                agent.epsilon = 0
                # 학습 수행
                loss_critic, loss_actor, maxQ = agent.train_model()
                loss_critic_list.append(loss_critic)
                loss_actor_list.append(loss_actor)
                max_Q_list.append(maxQ)

                # 타겟 네트워크 업데이트
                agent.soft_update_target()

            # 네트워크 모델 저장
            if step % config.save_step == 0 and step != 0 and train_mode:
                agent.save_model(config.load_model, train_mode)

        reward_list.append(episode_rewards)
        episode += 1

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
        if episode % config.print_episode == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss_critic: {:.4f}/ loss_actor: {:.4f}/ maxQ: {:.2f}".format
                  (step, episode, np.mean(reward_list), np.mean(loss_critic_list), np.mean(loss_actor_list), np.mean(max_Q_list)))
            if train_mode:
                agent.write_scalar(np.mean(loss_critic_list), np.mean(loss_actor_list), np.mean(reward_list), np.mean(max_Q_list), episode)

            reward_list = []
            loss_critic_list = []
            loss_actor_list = []
            max_Q_list = []

    agent.save_model(config.load_model, train_mode)
    env.close()

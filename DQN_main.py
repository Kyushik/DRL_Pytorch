# Import Libraries
import numpy as np
import random
import time
from collections import deque
from mlagents.envs import UnityEnvironment

import torch
import torch.optim as optim

import config
from agent import DQNAgent
import model

# Main function
if __name__ == '__main__':
    # set unity environment path (file_name)
    env = UnityEnvironment(file_name=config.env_name)

    # setting brain for unity
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    train_mode = config.train_mode

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_ = model.DQN(model_name="main", action_size=config.action_size,
    #                         in_channel=config.state_size[2]*config.stack_frame).to(device)
    # model_target_ = model.DQN(model_name="target", action_size=config.action_size,
    #                         in_channel=config.state_size[2]*config.stack_frame).to(device)
    model_ = model.DuelingDQN(model_name="main", action_size=config.action_size,
                            in_channel=config.state_size[2]*config.stack_frame).to(device)
    model_target_ = model.DuelingDQN(model_name="target", action_size=config.action_size,
                            in_channel=config.state_size[2]*config.stack_frame).to(device)
    optimizer = optim.Adam(model_.parameters(), lr=config.learning_rate)

    # Define Q_Agent class as agent
    agent = DQNAgent(model_, model_target_, optimizer, device)

    step = 0
    episode = 0
    rewards = []
    losses = []

    # Reset Unity environment and set the train mode according to the environment setting (env_config)
    env_info = env.reset(train_mode=train_mode, config=config.env_config)[default_brain]

    # Game loop
    while step < config.run_step + config.test_step:
        # Initialize state, episode_rewards, done
        obs = 255 * np.array(env_info.visual_observations[0]) # (1*80*80*3)
        obs = np.transpose(obs, (0, 3, 1, 2)) # (1*3*80*80)
        episode_rewards = 0
        done = False

        for i in range(config.skip_frame*config.stack_frame):
            # agent.obs_set.append(np.zeros([config.state_size[0], config.state_size[1], config.state_size[2]]))
            agent.obs_set.append(obs)

        state = agent.skip_stack_frame(obs)

        # loop for each episode
        while not done:
            if step == config.run_step:
                train_mode = False
                env_info = env.reset(train_mode=train_mode)[default_brain]

            # Decide action and apply the action to the Unity environment
            action = agent.get_action(state)
            env_info = env.step(action)[default_brain]

            # Get next state, reward, done information
            next_obs  = 255 * np.array(env_info.visual_observations[0])
            next_obs = np.transpose(next_obs, (0, 3, 1, 2))
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            next_state = agent.skip_stack_frame(next_obs)

            # save data to replay memory if it's train mode
            if train_mode:
                agent.append_memory(state, action, reward, next_state, done)
            else:
                time.sleep(0.0)
                agent.epsilon = 0.0

            # update state
            state = next_state
            step += 1

            if step > config.start_train_step and train_mode:
                # Decrease Epsilon
                if agent.epsilon > config.epsilon_min:
                    agent.epsilon -= 1 / (config.run_step - config.start_train_step)

                # perform training
                loss = agent.train_model_DQN(done)
                # loss = agent.train_model_doubleDQN(done)
                losses.append(loss)

                # update target network
                if step % (config.target_update_step) == 0:
                    agent.update_target()

            # save network model
            if step % config.save_step == 0 and step != 0:
                agent.save_model()

        rewards.append(episode_rewards)
        episode += 1

        # Print Progress
        if episode % config.print_episode == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon))

            agent.writer_tb(loss=np.mean(losses), reward=np.mean(rewards), episode=episode)
            rewards = []
            losses = []

    agent.save_model()
    env.close()

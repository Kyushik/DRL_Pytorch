# Import Libraries
import numpy as np
import random
import time
import datetime
from collections import deque
from mlagents.envs import UnityEnvironment

import agent 
import config
        
# Main function
if __name__ == '__main__':
    # set unity environment path (file_name)
    env = UnityEnvironment(file_name=config.env_name)

    # setting brain for unity 
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    train_mode = config.train_mode
    
    agent = agent.DQNAgent()

    step = 0
    episode = 0
    reward_list = []
    loss_list = []

    # Reset Unity environment and set the train mode according to the environment setting (env_config)  
    env_info = env.reset(train_mode=train_mode, config=config.env_config)[default_brain]

    # Game loop 
    while step < config.run_step + config.test_step:         
        # Initialize state, episode_rewards, done 
        obs = 255 * np.array(env_info.visual_observations[0])
        obs = np.transpose(obs, (0, 3, 1, 2))
        episode_rewards = 0
        done = False

        for i in range(config.skip_frame*config.stack_frame):
            agent.obs_set.append(obs)
        
        state = agent.skip_stack_frame(obs)

        # loop for an episode
        while not done:
            if step == config.run_step:
                train_mode = False
                env_info = env.reset(train_mode=train_mode)[default_brain]

            # Decide action and apply the action to the Unity environment 
            action = agent.get_action(state)
            env_info = env.step(action)[default_brain]

            # Get next state, reward, done information 
            next_obs = 255 * np.array(env_info.visual_observations[0])
            next_obs = np.transpose(next_obs, (0, 3, 1, 2))
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            next_state = agent.skip_stack_frame(next_obs)

            # Save data in replay memory while train mode 
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.0) 
                agent.epsilon = 0.0

            # Update state information 
            state = next_state
            step += 1

            if step > config.start_train_step and train_mode:
                # Epsilon 감소 
                if agent.epsilon > config.epsilon_min:
                    agent.epsilon -= 1 / (config.run_step - config.start_train_step)

                # 학습 수행 
                loss = agent.train_model(done)
                loss_list.append(loss)

                # 타겟 네트워크 업데이트 
                if step % (config.target_update_step) == 0:
                    agent.update_target()

            # 네트워크 모델 저장 
            if step % config.save_step == 0 and step != 0:
                agent.save_model()

        reward_list.append(episode_rewards)
        episode += 1

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        if episode % config.print_episode == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (step, episode, np.mean(reward_list), np.mean(loss_list), agent.epsilon))
            # agent.Write_Summray(np.mean(reward_list), np.mean(loss_list), episode)
            reward_list = []
            loss_list = []

    agent.save_model()
    env.close()





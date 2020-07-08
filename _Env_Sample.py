# Import Libraries
import numpy as np
import random
import time
from collections import deque
from mlagents.envs import UnityEnvironment

# Parameter Setting 
state_size = 6
action_size = 4 

run_step = 10000

print_episode = 10

train_mode = False 

# Environment Path 
game = "Pong"
env_name = "./env/" + game + "/Windows/" + game

def get_action():
    # select random action
    return np.random.randint(0, action_size)
        
# Main function
if __name__ == '__main__':
    # set unity environment path (file_name)
    env = UnityEnvironment(file_name=env_name)

    # setting brain for unity 
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    step = 0
    episode = 0
    reward_list = []

    # Reset Unity environment and set the train mode according to the environment setting   
    env_info = env.reset(train_mode=train_mode)[default_brain]

    # Game loop 
    while step < run_step:         
        # Initialize state, episode_rewards, done 
        state = str(env_info.vector_observations[0])
        episode_rewards = 0
        done = False

        # loop for an episode
        while not done:
            if step == run_step:
                train_mode = False
                env_info = env.reset(train_mode=train_mode)[default_brain]

            # Decide action and apply the action to the Unity environment 
            action = get_action()
            env_info = env.step(action)[default_brain]

            # Get next state, reward, done information 
            next_state = 255 * np.array(env_info.vector_observations[0])
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            time.sleep(0.02) 

            # Update state info 
            state = next_state
            step += 1

        reward_list.append(episode_rewards)
        episode += 1

        # Print Progress
        if episode != 0 and episode % print_episode == 0:
            print("Step: {} / Episode: {} / Mean Rewards: {:.3f}".format(step, episode, np.mean(reward_list)))
            reward_list = []

    env.close()





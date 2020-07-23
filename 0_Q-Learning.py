# Import Libraries
import numpy as np
import random
import time
from collections import deque
from mlagents.envs import UnityEnvironment

# Parameter Setting 
state_size = 6
action_size = 4 

load_model = False
train_mode = True

discount_factor = 0.9
learning_rate = 0.1

run_step = 100000
test_step = 2500

print_episode = 100

epsilon_init = 1.0
epsilon_min = 0.1

# GridWorld Environment Setting (Gameboard size=4)
env_config = {"gridSize": 4}

# Environment Path 
game = "GridWorld"
env_name = "./env/" + game + "/Windows/" + game

# Q_Agent class -> Functions for Q-Learining are defined 
class Q_Agent():
    def __init__(self):
        self.Q_table = {}
        self.epsilon = epsilon_init

    # If there is no state information in Q-table, initialize Q-table
    def init_Q_table(self, state):
        if state not in self.Q_table.keys():
            self.Q_table[state] = np.zeros(action_size)

    # Decide action according to Epsilon greedy 
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # select random action
            return np.random.randint(0, action_size)
        else:
            self.init_Q_table(state)

            # select action with respect to the Q-table
            predict = np.argmax(self.Q_table[state])
            return predict

    # Perform Training
    def train_model(self, state, action, reward, next_state, done):
        self.init_Q_table(state)
        self.init_Q_table(next_state)

        # calculate target value and update Q-Table
        target = reward + discount_factor * np.max(self.Q_table[next_state])
        Q_val = self.Q_table[state][action]

        if done:
            self.Q_table[state][action] = reward
        else:
            self.Q_table[state][action] = (1-learning_rate) * Q_val + learning_rate * target
        
# Main function
if __name__ == '__main__':
    # set unity environment path (file_name)
    env = UnityEnvironment(file_name=env_name)

    # setting brain for unity 
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # Define Q_Agent class as agent 
    agent = Q_Agent()

    step = 0
    episode = 0
    reward_list = []

    # Reset Unity environment and set the train mode according to the environment setting (env_config)  
    env_info = env.reset(train_mode=train_mode, config=env_config)[default_brain]

    # Game loop 
    while step < run_step + test_step:         
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
            action = agent.get_action(state)
            env_info = env.step(action)[default_brain]

            # Get next state, reward, done information 
            next_state = str(env_info.vector_observations[0])
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            # Update the Q-table if it is train mode 
            if train_mode:
                # Decrease Epsilon 
                if agent.epsilon > epsilon_min:
                    agent.epsilon -= 1 / run_step

                agent.train_model(state, action, reward, next_state, done)
            else:
                time.sleep(0.02) 
                agent.epsilon = 0.0

            # Update state info 
            state = next_state
            step += 1

        reward_list.append(episode_rewards)
        episode += 1

        # Print Progress
        if episode != 0 and episode % print_episode == 0:
            print("Step: {} / Episode: {} / Epsilon: {:.3f} / Mean Rewards: {:.3f}".format(step, episode, agent.epsilon, np.mean(reward_list)))
            reward_list = []

    env.close()
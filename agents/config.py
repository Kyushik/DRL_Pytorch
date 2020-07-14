# Config
import datetime

# Parameter Setting
state_size = [80,80,3]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

skip_frame = 1
stack_frame = 1

run_step = 1000000
test_step = 50000

start_train_step = 20000

target_update_step = 10000
print_episode = 100
save_step = 50000

epsilon_init = 1.0
epsilon_min = 0.1

# GridWorld Environment Setting (Gameboard size=4)
env_config = {"gridSize": 4}

# Environment Path
game = "GridWorld"
env_name = "./env/" + game + "/Windows/" + game

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# 모델 저장 및 불러오기 경로
save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
load_path = "../saved_models/" + game + "/20200221-10-30-27_DQN/model.pth"

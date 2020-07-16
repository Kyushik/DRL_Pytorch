# Config
import datetime

state_size = [80,80,3]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000

discount_factor = 0.9
learning_rate = 0.00025

skip_frame = 4
stack_frame = 4

start_train_step = 10000
run_step = 500000
test_step = 2500

target_update_step = int(run_step/100)
print_episode = 10
save_step = 10000

epsilon_init = 1.0
epsilon_min = 0.1

# Environment Setting
env_config = {'gridSize':3}
# env_config = {}

# Environment Path
game = "GridWorld"
env_name = "./env/" + game + "/Windows/" + game

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + game + "/" + date_time + "_DeulingDQN"
load_path = "./saved_models/" + game + "/20200713-18-46-18_DQN/model.pth"

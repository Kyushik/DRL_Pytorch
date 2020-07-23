# Config
import datetime
import torch

state_size = [80,80,3]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 100000

discount_factor = 0.99
learning_rate = 0.0001

skip_frame = 4
stack_frame = 4

start_train_step = 10000
run_step = 250000
test_step = 25000

target_update_step = int(run_step/100)
print_episode = 10
save_step = 10000

epsilon_init = 1.0
epsilon_min = 0.1

# Parameters for Curiosity-driven Exploration
beta = 0.2
lamb = 1.0
eta = 0.01
extrinsic_coeff = 1.0
intrinsic_coeff = 0.01

# Environment Setting
env_config = {'gridSize':3}
# env_config = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Environment Path
game = "GridWorld"
env_name = "./env/" + game + "/Windows/" + game

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + game + "/" + date_time
load_path = "./saved_models/" + game + "/20200721-20-44-39_DoubleDQN/model.pth"

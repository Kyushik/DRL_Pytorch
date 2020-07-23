# Config
import datetime

state_size = [40,80,1]
action_size = 3

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 100000

discount_factor = 0.99
learning_rate = 0.0001

skip_frame = 4
stack_frame = 4

start_train_step = 10000
run_step = 500000
test_step = 25000

target_update_step = int(run_step/100)
print_episode = 10
save_step = 10000

epsilon_init = 1.0
epsilon_min = 0.1

# Environment Setting
# env_config = {'gridSize':3}
env_config = {}


""" C51 """
atoms = 51
vmin = -10
vmax = 10

# Environment Path
game = "Pong"
env_name = "./env/" + game + "/Linux/" + game

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + game + "/" + date_time
load_path = "./saved_models/" + game + "/20200722-00-28-43_NoisyDQN/model.pth"

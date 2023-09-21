import torch
import datetime


class DataForDQN:
    state_size = 48 # state size of IMRLEnv : 3*12(traffic light one-hot encoding) + 12(average waiting time section)
    action_size = 13 # action size of IMRLEnv : 0 ~ 11(traffic light number), 12(no-op)

    # This section need to be parameterized
    load_model = False
    train_mode = True

    batch_size = 32
    mem_maxlen = 10000
    discount_factor = 0.9
    learning_rate = 0.00025

    run_step = 500000
    test_step = 0
    train_start_step = 300
    target_update_step = 300

    print_interval = 1
    save_interval = 10

    is_epsilon = True

    # Using epsilon-greedy policy. Future work should add boltzmann policy
    epsilon_eval = 0.5
    epsilon_init = 0.9
    epsilon_min = 0.1
    explore_step = run_step*0.5
    epsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0

    # boltzmann policy parameters

    '''
    # these are not needed in this code
    VISUAL_OBS = 0
    GOAL_OBS = 1
    VECTOR_OBS = 2
    OBS = VECTOR_OBS
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"../saved_models/DQN/{date_time}"
    load_path = f"../saved_models/DQN/{date_time}" # should modified to get certain path to execute

def dqn_data_print(dqn_data):
    print("... Data Fro DQN ...")
    print("state size : {}".format(dqn_data.state_size))
    print("action size : {}".format(dqn_data.action_size))
    print("batch size : {}".format(dqn_data.batch_size))
    print("experience memory max size : {}".format(dqn_data.mem_maxlen))
    print("discount factor : {}".format(dqn_data.discount_factor))
    print("learning rate : {}".format(dqn_data.learning_rate))
    print("run step : {}".format(dqn_data.run_step))
    print("test step : {}".format(dqn_data.test_step))
    print("train start step : {}".format(dqn_data.train_start_step))
    print("target update step : {}".format(dqn_data.target_update_step))
    print("print interval(episode) : {}".format(dqn_data.print_interval))
    print("save interval(episode) : {}".format(dqn_data.save_interval))
    print("epsilon greedy policy : {}".format(dqn_data.is_epsilon))
    print("epsilon init : {} / epsilon min : {} / epsilon decrese step : {}".format(dqn_data.epsilon_init, 
                                                                                    dqn_data.epsilon_min, dqn_data.explore_step))
    print("torch device : {}".format(dqn_data.device))
    print("save path : {}".format(dqn_data.save_path))

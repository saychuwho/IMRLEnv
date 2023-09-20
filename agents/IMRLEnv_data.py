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

    def data_print(self):
        print("... Data Fro DQN ...")
        print("state size : {}".format(self.state_size))
        print("action size : {}".format(self.action_size))
        print("batch size : {}".format(self.batch_size))
        print("experience memory max size : {}".format(self.mem_maxlen))
        print("discount factor : {}".format(self.discount_factor))
        print("learning rate : {}".format(self.learning_rate))
        print("run step : {}".format(self.run_step))
        print("test step : {}".format(self.test_step))
        print("train start step : {}".format(self.train_start_step))
        print("target update step : {}".format(self.target_update_step))
        print("print interval(episode) : {}".format(self.print_interval))
        print("save interval(episode) : {}".format(self.save_interval))
        print("epsilon greedy policy : {}".format(self.is_epsilon))
        print("epsilon init : {} / epsilon min : {} / epsilon decrese step : {}".format(self.epsilon_init, 
                                                                                        self.epsilon_min, self.explore_step))
        print("torch device : {}".format(self.device))
        print("save path : {}".format(self.save_path))

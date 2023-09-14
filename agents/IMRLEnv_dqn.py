'''
dqn for IMRLEnv
main code is from book "파이토치와 유니티 ML-Agents로 배우는 강화학습" DQNAgent.py

original code is made for visual observation. IMRLEnv uses vector obs, so I modified that points

This code uses epsilon-greedy policy. I will make this work with boltzmann policy soon
'''

import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class DataForDQN:
    state_size = 48 # state size of IMRLEnv : 3*12(traffic light one-hot encoding) + 12(average waiting time)
    action_size = 13 # action size of IMRLEnv : 0 ~ 11(traffic light number), 12(no-op)

    # This section need to be parameterized
    load_model = False
    train_mode = True

    batch_size = 32
    mem_maxlen = 10000
    discount_factor = 0.9
    learning_rate = 0.00025

    run_step = 50000
    test_step = 5000
    train_start_step = 500
    target_update_step = 500

    print_interval = 10
    save_interval = 100

    # Using epsilon-greedy policy. Future work should add boltzmann policy
    epsilon_eval = 0.5
    epsilon_init = 0.1
    epsilon_min = 0.1
    explore_step = run_step*0.8
    epsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0

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


'''
changed DQN(Q function) into Linear. Because IMRLEnv has vector observation. not visual obs
'''
class Q_function(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Q_function, self).__init__(**kwargs)
        self.dqn_data = DataForDQN
        self.d1 = torch.nn.Linear(self.dqn_data.state_size, 128)
        self.d2 = torch.nn.Linear(128,128)
        self.q = torch.nn.Linear(128,self.dqn_data.action_size)
    
    def forward(self,x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.q(x)
    

class DQNAgent:
    def __init__(self):
        self.dqn_data = DataForDQN
        self.network = Q_function().to(self.dqn_data.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.dqn_data.learning_rate)
        self.memory = deque(maxlen=self.dqn_data.mem_maxlen)
        self.epsilon = self.dqn_data.epsilon_init
        self.writer = SummaryWriter(self.dqn_data.save_path)

        if self.dqn_data.load_model == True:
            print(f"... Load Model from {self.dqn_data.load_path}/ckpt ...")
            checkpoint = torch.load(self.dqn_data.load_path+'/ckpt', map_location=self.dqn_data.device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.loat_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # using epsilon-greedy first. future work should add boltzmann policy
    def get_action(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.dqn_data.epsilon_eval

        if epsilon > random.random():
            action = np.random.randint(0, self.dqn_data.action_size, size=(1,1,1))
        else:
            q = self.network(torch.FloatTensor(state).to(self.dqn_data.device))
            action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        return action
    
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, self.dqn_data.batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.dqn_data.device), [state, action, reward, next_state, done])

        eye = torch.eye(self.dqn_data.action_size).to(self.dqn_data.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state)*one_hot_action).sum(1,keepdims=True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1 - done) * self.dqn_data.discount_factor)
        
        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.dqn_data.epsilon_min, self.epsilon - self.dqn_data.epsilon_delta)

        return loss.item()
    
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self):
        print(f"... Save Model to {self.dqn_data.save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, self.dqn_data.save_path+'/ckpt')
    
    def write_summary(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)


def main(Env: UnityEnvironment):
    dqn_data = DataForDQN

    engine_configuration_channel = EngineConfigurationChannel()
    env = Env
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    # time scale should be setted as env setting
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0) 
    dec, term = env.get_steps(behavior_name)

    agent = DQNAgent()

    losses, scores, episode, score = [], [], 0, 0

    train_mode = dqn_data.train_mode

    print("... DQN STARTS ...")

    
    for step in range(dqn_data.run_step + dqn_data.test_step):
        if step == dqn_data.run_step:
            if train_mode:
                agent.save_model()
            print("... TEST START ...")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
        
        state = dec.obs # obs contains vector obs
        action = agent.get_action(state, train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action[0])

        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs if done else dec.obs
        score += reward[0]

        if train_mode:
            agent.append_sample(state, action[0], reward, next_state, [done])

        if train_mode and step > max(dqn_data.batch_size, dqn_data.train_start_step):
            loss = agent.train_model()
            losses.append(loss)

            if step % dqn_data.target_update_step == 0:
                agent.update_target()
        
        if done:
            episode += 1
            scores.append(score)
            score = 0

            if episode % dqn_data.print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")
                
            if train_mode and episode % dqn_data.save_interval == 0:
                agent.save_model()
    
    env.close()


if __name__ == '__main__':
    main()
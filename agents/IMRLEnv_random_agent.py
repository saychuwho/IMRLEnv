# random agent for IMRLEnv
# code base is from book "learning Reinforcement Learning using pytorch and unity ml-agent"

from mlagents_envs.environment import UnityEnvironment
# from IMRLEnv_main import CustomSideChannel

def main(env : UnityEnvironment):
    # env = UnityEnvironment(file_name='../IMRLEnv/IMRLEnv')

    #channel.send_string("0,-1")
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    print(f'name of behavior:{behavior_name}')
    spec = env.behavior_specs[behavior_name]

    for ep in range(10):

        print("episode : {}".format(ep+1))

        # SideChannel logic
        #channel.send_string("0,{}".format(ep+1))
        env.reset()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        tracked_agent = -1
        done = False
        ep_rewards = 0

        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # random_action generates random ActionTuple for a number(gets by n_agent: int) of agents
            action = spec.action_spec.random_action(len(decision_steps))

            env.set_actions(behavior_name, action)

            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if tracked_agent in decision_steps:
                ep_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                ep_rewards += terminal_steps[tracked_agent].reward
                done = True

        print(f'total reward for ep {ep} is {ep_rewards}')

    env.close()


if __name__ == '__main__':
    main()


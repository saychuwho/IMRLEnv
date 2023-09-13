'''
# sets env with SideChannel
# modules for SideChannel. not used
from mlagents_envs.side_channel import IncomingMessage

from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid
'''

import IMRLEnv_random_agent as random_agent
import IMRLEnv_dqn as dqn_agent
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import argparse


'''
# class for SideChannel. not used
class CustomSideChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("718dcd44-b859-4103-b932-837d056de9d2"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
'''

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--platform', type=str, default='linux-server', help='Select platform running IMRLEnv')
    parser.add_argument('--agent', type=str, default='random_agent', help='select agent')

    args = parser.parse_args()
    
    env_file_name = '../IMRLEnv_{}/IMRLEnv'.format(args.platform)
    env = UnityEnvironment(file_name=env_file_name)
    
    if args.agent == 'random_agent':
        random_agent.main(env)
    elif args.agent == 'dqn_agent':
        dqn_agent.main(env)


if __name__ == '__main__':
    main()
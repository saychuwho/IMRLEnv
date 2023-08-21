# sets env with SideChannel
'''
# modules for SideChannel. not used
from mlagents_envs.side_channel import IncomingMessage

from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid
'''

import IMRLEnv_random_agent as RandomAgent
from mlagents_envs.environment import UnityEnvironment
import numpy as np



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
    env = UnityEnvironment(file_name='../IMRLEnv_Windows/IMRLEnv')

    RandomAgent.main(env)


if __name__ == '__main__':
    main()
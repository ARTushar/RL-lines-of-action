from enum import Enum, auto

import numpy as np

from environments.lines_of_action.lac import LACEnv
from utils.agent import BotAgent, RandomAgent


class OpponentType(Enum):
    RANDOM = auto()
    BOT = auto()


class SelfPlayEnv(LACEnv):
    def __init__(self, opponent_type: OpponentType):
        super(SelfPlayEnv, self).__init__()
        self.opponent_type = opponent_type
        self.opponent_player_num = None
        self.agent_player_num = None
        self.opponent_agent = None

    def setup_opponents(self):
        self.agent_player_num = np.random.choice(self.n_players) + 1
        self.opponent_player_num = 1 if self.agent_player_num == 2 else 2

        if self.opponent_type == OpponentType.BOT:
            self.opponent_agent = BotAgent('../bots/player2', self.opponent_player_num)
        elif self.opponent_agent == OpponentType.RANDOM:
            self.opponent_agent = RandomAgent(self.opponent_player_num)

    def reset(self):
        if self.engine.current_player != self.agent_player_num:
            self.continue_game()
        self.setup_opponents()

        return super(SelfPlayEnv, self).reset()

    def continue_game(self):
        done, info, observation = None, None, None
        if self.opponent_type == OpponentType.BOT:
            moves = self.opponent_agent.get_move(self.engine.board)
            action = self.move_to_action(((moves[0], moves[1]), (moves[2], moves[3])))
            observation, reward, done, info = super(SelfPlayEnv, self).step(action)

        elif self.opponent_type == OpponentType.RANDOM:
            action = self.opponent_agent.choose_action(self, True)
            observation, reward, done, info = super(SelfPlayEnv, self).step(action)

        return observation, self.engine.opponent_reward, done, info

    def step(self, action):
        observation, reward, done, info = super(SelfPlayEnv, self).step(action)

        if not done:
            observation, reward, done, info = self.continue_game()
            reward = self.engine.opponent_reward

        return observation, reward, done, info

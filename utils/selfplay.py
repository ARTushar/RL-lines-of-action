from enum import Enum, auto

import numpy as np
from stable_baselines3.common.env_checker import check_env

from environments.lines_of_action.lac import LACEnv
from utils.agent import BotAgent, RandomAgent
from utils.helpers import load_best_model

class OpponentType(Enum):
    RANDOM = auto()
    BOT = auto()
    PREV_BEST = auto()
    PREV_RANDOM = auto()
    BASE = auto()


class SelfPlayEnv(LACEnv):
    def __init__(self, opponent_type: OpponentType, verbose: int = 1):
        super(SelfPlayEnv, self).__init__(verbose=verbose)
        self.opponent_type = opponent_type
        self.verbose = verbose
        self.opponent_player_num = None
        self.agent_player_num = None
        self.opponent_agent = None
        # self.setup_opponents()

    def setup_opponents(self):
        self.agent_player_num = np.random.choice(self.n_players) + 1
        self.opponent_player_num = -1 if self.agent_player_num == 2 else 1
        self.agent_player_num = -1 if self.opponent_player_num == 1 else 1
        if self.verbose >= 1:
            print("Agent player: ", self.agent_player_num)
            print("Opponent player: ", self.opponent_player_num)

        if self.opponent_type == OpponentType.BOT:
            self.opponent_agent = BotAgent('bots/player2', self.opponent_player_num)
        elif self.opponent_type == OpponentType.RANDOM:
            self.opponent_agent = RandomAgent(self.opponent_player_num)
        elif self.opponent_type == OpponentType.PREV_BEST:
            self.opponent_agent = load_best_model(self)

    def reset(self):
        observation = super(SelfPlayEnv, self).reset()
        self.setup_opponents()
        if self.engine.current_player != self.agent_player_num:
            observation, _, _, _ = self.continue_game()

        return observation

    def continue_game(self):
        action = self.opponent_agent.choose_action(self, True)
        observation, reward, done, info = super(SelfPlayEnv, self).step(action)
        return observation, self.engine.opponent_reward, done, info

    def step(self, action):
        observation, reward, done, info = super(SelfPlayEnv, self).step(action)

        if not done:
            observation, reward, done, info = self.continue_game()
            reward = self.engine.opponent_reward

        return observation, reward, done, info


def test_self_play_env():
    self_play = SelfPlayEnv(OpponentType.BOT)
    check_env(self_play)


if __name__ == '__main__':
    test_self_play_env()

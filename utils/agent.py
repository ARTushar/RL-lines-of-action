from abc import ABC, abstractmethod
from subprocess import Popen, PIPE
from environments.lines_of_action.lac import LACEnv
import gym
import numpy as np


class Agent(ABC):
    def __init__(self, player_no: int):
        self.player_no = player_no


    @abstractmethod
    def choose_action(self, env: gym.Env, choose_best_action: bool):
        pass

    @staticmethod
    def sample_action(action_probs):
        action = np.random.choice(len(action_probs), p=action_probs)
        return action


class RandomAgent(Agent):
    def __init__(self, player_no):
        super(RandomAgent, self).__init__(player_no)

    def choose_action(self, env: gym.Env, choose_best_action: bool):
        action_probs = np.array(env.get_random_action())
        action = np.argmax(action_probs)
        if not choose_best_action:
            action = self.sample_action(action_probs)

        return action


class BotAgent(Agent):
    def __init__(self, filename, player_no):
        super(BotAgent, self).__init__(player_no)
        self.filename = filename,
        self.process = self.open_minmax_bot(filename, player_no)

    def choose_action(self, env: gym.Env, choose_best_action: bool=True):
        board = env.engine.board
        board_str = self.convert_board(board)
        print(board_str, file=self.process.stdin, flush=True)
        moves = self.process.stdout.readline()
        moves = [int(x) for x in moves[:-1].split()]
        action = env.move_to_action(((moves[0], moves[1]), (moves[2], moves[3])))
        return action

    @staticmethod
    def convert_board(board):
        board_str = ''
        for row in board:
            for val in row:
                board_str += BotAgent.get_first_char(val) + ' '
            board_str += '\n'
        return board_str

    @staticmethod
    def get_first_char(val):
        if val == -1:
            return 'b'
        elif val == 1:
            return 'r'
        return 'e'

    @staticmethod
    def open_minmax_bot(filename, player_no, board_size=8):
        if player_no == 1:
            player_type = 'b'
        else:
            player_type = 'r'
        command = [filename, player_type, str(board_size)]
        return Popen(command, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize=1)

# TODO: Create a model agent


def test_random_agent():
    agents = [RandomAgent(1), RandomAgent(2)]
    env = LACEnv()

    env.render()
    game_done = False
    while not game_done:
        for agent in agents:
            action = agent.choose_action(env, True)
            print('move: ', env.action_to_move(action))
            _, reward, done, _ = env.step(action)
            env.render()
            if done:
                game_done = True
                break


def test_bot_agent():
    agents = [BotAgent('../bots/player1', 1), BotAgent('../bots/player2', 2)]
    env = LACEnv()
    env.render()
    game_done = False
    while not game_done:
        for agent in agents:
            action = agent.choose_action(env)
            _, reward, done, _ = env.step(action)
            env.render()
            if done:
                game_done = True
                break


if __name__ == '__main__':
    np.random.seed(1)
    # test_random_agent()
    test_bot_agent()

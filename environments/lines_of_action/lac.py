import gym
import numpy as np
import os
import platform
from typing import Tuple
from stable_baselines3.common.env_checker import check_env
from environments.lines_of_action.game import Game

MOVE = Tuple[Tuple, Tuple]
clear_command = 'cls' if platform.system() == 'Windows' else 'clear'
clear = lambda: os.system(clear_command)


class LACEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    n_players = 2
    grid_len = 8

    def __init__(self, verbose: bool = False, manual: bool = False):
        super(LACEnv, self).__init__()
        self.name = 'Lines of Action'
        self.manual = manual
        self.verbose = verbose
        self.num_squares = self.grid_len * self.grid_len
        self.grid_shape = (self.grid_len, self.grid_len)
        self.engine = Game(self.grid_len)

        self.action_space: gym.spaces.Space = gym.spaces.Discrete(self.num_squares * self.num_squares)
        self.observation_space: gym.spaces.Space = gym.spaces.Box(-1, 1, self.grid_shape + (13, ))

    def step(self, action):
        move: MOVE = self.action_to_move(action)
        reward = self.engine.step(move[0], move[1])
        observation = self.create_observation(self.engine.board, self.engine.get_all_current_player_moves())
        return observation, reward, self.engine.done, {'state': self.engine.board}

    def reset(self):
        self.engine.reset_game()
        return self.create_observation(self.engine.board, self.engine.get_all_current_player_moves())

    def render(self, mode="human"):
        clear()
        print('--------------------------------------------------')
        if self.engine.done:
            print("GAME OVER")
            winner = 'player 1' if self.engine.winner == -1 else 'Player 2'
            print("Winner: ", winner)
        else:
            current_player = '1' if self.engine.current_player == -1 else '2'
            print("It is Player {}'s turn".format(current_player))
        print('--------------------------------------------------')
        self.engine.print_board()
        print('--------------------------------------------------')

    def get_random_action(self):
        all_valid_moves = self.engine.get_all_current_player_moves()
        random_pos = np.random.randint(len(all_valid_moves))
        selected = all_valid_moves[random_pos]
        move_from = selected[0]
        random_to = np.random.randint(len(selected[1]))
        move_to = selected[1][random_to]
        action = self.move_to_action((move_from, move_to))
        return self.create_action_probs(action)

    @staticmethod
    def create_action_probs(action):
        n = pow(LACEnv.grid_len, 4) - 1
        action_prob = 0.9
        other_prob = 0.1 / n
        action_probs = [other_prob] * n
        action_probs[action]  = action_prob
        return action_probs

    @staticmethod
    def move_to_action(move: MOVE) -> int:
        grid_len = LACEnv.grid_len
        total_squares = grid_len * grid_len

        selected = move[0][0] * grid_len + move[0][1]
        target = move[1][0] * grid_len + move[1][1]
        return selected * total_squares + target

    @staticmethod
    def action_to_move(action: int) -> MOVE:
        grid_len = LACEnv.grid_len
        total_squares = grid_len * grid_len

        selected, target = action // total_squares, action % total_squares
        sel_row, sel_col = selected // grid_len, selected % grid_len
        tar_row, tar_col = target // grid_len, target % grid_len

        return (sel_row, sel_col), (tar_row, tar_col)

    @staticmethod
    def create_observation(board, all_valid_moves):
        all_frames = [np.array(board, dtype='float32')]
        for pos, valid_moves in all_valid_moves:
            all_frames.append(LACEnv.get_valid_move_frame(pos, valid_moves))

        for i in range(12 - len(all_valid_moves)):
            all_frames.append(np.zeros((len(board), len(board)), dtype='float32'))

        final_frame = np.stack(all_frames, axis=-1)
        # print('shape of frame: ', final_frame.shape)
        return final_frame

    @staticmethod
    def get_valid_move_frame(pos, valid_moves) -> np.ndarray:
        ret = np.zeros((LACEnv.grid_len, LACEnv.grid_len), dtype='float32')
        ret[pos[0], pos[1]] = -1  # the moving checker position
        for move in valid_moves:
            ret[move[0], move[1]] = 1  # valid cells where the checker can be moved

        return ret


def test_env():
    env = LACEnv()
    check_env(env)


if __name__ == '__main__':
    test_env()

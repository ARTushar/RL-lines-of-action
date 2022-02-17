from enum import Enum
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


class MoveDirection(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    LEFT_UP = 4
    LEFT_DOWN = 5
    RIGHT_UP = 6
    RIGHT_DOWN = 7
    INVALID = 8


class LACEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    n_players = 2
    grid_len = 8
    valid_cell_value = 0.5
    total_valid_directions = 8

    def __init__(self, verbose: int = 1):
        super(LACEnv, self).__init__()
        self.name = 'Lines of Action'
        self.verbose = verbose
        self.num_squares = self.grid_len * self.grid_len
        self.grid_shape = (self.grid_len, self.grid_len)
        self.engine = Game(self.grid_len, verbose=verbose)

        self.action_space: gym.spaces.Space = gym.spaces.Discrete(self.num_squares * self.total_valid_directions)
        self.observation_space: gym.spaces.Space = gym.spaces.Box(-1, 1, (13,) + self.grid_shape)

    def step(self, action):
        move: MOVE = self.action_to_move(action, self.engine.board, self.engine.current_player)
        reward = self.engine.step(move[0], move[1])
        observation = self.create_observation(self.engine.board, self.engine.get_all_current_player_moves(), self.engine.current_player)
        return observation, reward, self.engine.done, {'state': self.engine.board}

    def reset(self):
        self.engine.reset_game()
        return self.create_observation(self.engine.board, self.engine.get_all_current_player_moves(), self.engine.current_player)

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
        action_probs[action] = action_prob
        return action_probs

    @staticmethod
    def move_to_direction(move: MOVE) -> MoveDirection:
        selected = move[0]
        target = move[1]
        is_up, is_down, is_left, is_right = False, False, False, False
        d_row = target[0] - selected[0]
        d_col = target[1] - selected[1]
        if d_row < 0:
            is_up = True
        elif d_row > 0:
            is_down = True
        if d_col < 0:
            is_left = True
        elif d_col > 0:
            is_right = True

        if not is_up and not is_down and not is_right and is_left:
            return MoveDirection.LEFT
        if not is_up and not is_down and is_right and not is_left:
            return MoveDirection.RIGHT
        if is_up and not is_down and not is_right and not is_left:
            return MoveDirection.UP
        if not is_up and is_down and not is_right and not is_left:
            return MoveDirection.DOWN
        if is_up and not is_down and not is_right and is_left:
            return MoveDirection.LEFT_UP
        if not is_up and is_down and not is_right and is_left:
            return MoveDirection.LEFT_DOWN
        if is_up and not is_down and is_right and not is_left:
            return MoveDirection.RIGHT_UP
        if not is_up and is_down and is_right and not is_left:
            return MoveDirection.RIGHT_DOWN
        return MoveDirection.INVALID

    @staticmethod
    def direction_to_move(selected: Tuple[int, int], direction: MoveDirection, board, current_player) -> MOVE:
        row_cells, col_cells, left_diagonal_cells, right_diagonal_cells = Game.calculate_total_cells(board, current_player)
        if direction is MoveDirection.LEFT:
            return selected, (selected[0], selected[1] - row_cells)
        if direction is MoveDirection.RIGHT:
            return selected, (selected[0], selected[1] + row_cells)
        if direction is MoveDirection.UP:
            return selected, (selected[0] - col_cells, selected[1])
        if direction is MoveDirection.DOWN:
            return selected, (selected[0] + col_cells, selected[1])
        if direction is MoveDirection.LEFT_UP:
            return selected, (selected[0] - left_diagonal_cells, selected[1] - left_diagonal_cells)
        if direction is MoveDirection.RIGHT_UP:
            return selected, (selected[0] - right_diagonal_cells, selected[1] + right_diagonal_cells)
        if direction is MoveDirection.LEFT_DOWN:
            return selected, (selected[0] + right_diagonal_cells, selected[1] - right_diagonal_cells)
        if direction is MoveDirection.RIGHT_DOWN:
            return selected, (selected[0] + left_diagonal_cells, selected[1] + left_diagonal_cells)

    @staticmethod
    def move_to_action(move: MOVE) -> int:
        grid_len = LACEnv.grid_len
        # total_squares = grid_len * grid_len

        selected = move[0][0] * grid_len + move[0][1]
        # target = move[1][0] * grid_len + move[1][1]
        direction = LACEnv.move_to_direction(move)
        assert direction is not MoveDirection.INVALID

        return selected * LACEnv.total_valid_directions + direction.value

    @staticmethod
    def action_to_move(action: int, board, current_player) -> MOVE:
        grid_len = LACEnv.grid_len
        # total_squares = grid_len * grid_len

        selected, direction = action // LACEnv.total_valid_directions, action % LACEnv.total_valid_directions
        direction = MoveDirection(direction)
        sel = selected // grid_len, selected % grid_len
        move = LACEnv.direction_to_move(sel, direction, board, current_player)
        return move

    @staticmethod
    def create_observation(board, all_valid_moves, current_player):
        all_frames = [np.array(board, dtype='float32')]
        for pos, valid_moves in all_valid_moves:
            all_frames.append(LACEnv.get_valid_move_frame(pos, valid_moves, current_player))

        for i in range(12 - len(all_valid_moves)):
            all_frames.append(np.zeros((len(board), len(board)), dtype='float32'))

        final_frame = np.stack(all_frames, axis=0)
        # print('shape of frame: ', final_frame.shape)
        return final_frame

    @staticmethod
    def get_valid_move_frame(pos, valid_moves, current_player) -> np.ndarray:
        ret = np.zeros((LACEnv.grid_len, LACEnv.grid_len), dtype='float32')
        ret[pos[0], pos[1]] = current_player  # the moving checker position
        for move in valid_moves:
            ret[move[0], move[1]] = LACEnv.valid_cell_value  # valid cells where the checker can be moved

        return ret


def test_env():
    env = LACEnv()
    check_env(env)


if __name__ == '__main__':
    test_env()

import sys

from environments.lines_of_action.lac import LACEnv
from utils.agent import ModelAgent


class Player:
    def __init__(self, player_type: str, board_size: int):
        self.player_type = player_type
        self.player_no = self.get_player_no()
        self.board_size = board_size
        self.env = LACEnv()
        self.model_agent = ModelAgent(self.player_no, self.env, model='best', verbose=0)
        self.prev_board_state = None
        self.observation = None
        self.current_board_state = None
        self.current_action = None
        self.current_move = None

        self.log_file = open('log_file.txt', 'w')

        self.set_initial_board_state()
        self.iterations = 0
        print('Playing as player :', self.player_no, file=self.log_file, flush=True)

    def print_agent_move(self):
        print("Iteration: ", self.iterations, file=self.log_file, flush=True)
        vals = [str(self.current_move[0][0]), str(self.current_move[0][1]), str(self.current_move[1][0]), str(self.current_move[1][1])]
        vals = ' '.join(vals)

        print(vals, file=self.log_file, flush=True)
        print(vals, flush=True)

    def get_player_no(self):
        if self.player_type == 'b':
            return 1
        return 2

    def set_initial_board_state(self):
        self.current_board_state = self.get_initial_board_state()
        print(self.current_board_state, file=self.log_file, flush=True)

        if self.player_no == 1:
            self.observation = self.env.reset()
            self.current_action = self.model_agent.choose_action(self.env, choose_best_action=True, mask_action=True, observation=self.observation)
            self.current_move = self.env.action_to_move(self.current_action)

    def step(self):
        board_str = ''
        for _ in range(self.board_size):
            board_str += input() + "\n"

        self.prev_board_state = self.current_board_state
        self.current_board_state = board_str

        if self.iterations == 0 and self.player_no == 1:
            self.iterations += 1
            self.observation, reward, done, _ = self.env.step(self.current_action)
            self.prev_board_state = self.current_board_state
            self.current_board_state = self.convert_board(self.env.engine.board)
            return
        print('board states', file=self.log_file, flush=True)
        print(self.prev_board_state, file=self.log_file, flush=True)
        print(self.current_board_state, file=self.log_file, flush=True)

        selected, target, player = self.board_states_to_move(self.prev_board_state, self.current_board_state)
        print("selected: ", selected, " type: ", type(selected[1]), file=self.log_file, flush=True)
        print("target: ", target, " type: ", type(target[1]), file=self.log_file, flush=True)

        action = self.env.move_to_action((selected, target))
        self.observation, reward, done, _ = self.env.step(action)
        self.prev_board_state = self.current_board_state
        self.current_board_state = self.convert_board(self.env.engine.board)

        self.current_action = self.model_agent.choose_action(self.env, choose_best_action=True, mask_action=True, observation=self.observation)
        self.current_move = self.env.action_to_move(self.current_action)
        self.observation, reward, done, _ = self.env.step(self.current_action)
        self.prev_board_state = self.current_board_state
        self.current_board_state = self.convert_board(self.env.engine.board)

        self.iterations += 1

    def get_initial_board_state(self):
        board_str = ''
        for i in range(self.board_size):
            if i == 0 or i == self.board_size-1:
                board_str += 'e b b b b b b e\n'
            else:
                board_str += 'r e e e e e e r\n'
        return board_str

    @staticmethod
    def convert_board(board):
        board_str = ''
        for row in board:
            for val in row:
                board_str += Player.get_first_char(val) + ' '
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
    def board_states_to_move(board1, board2):
        pos1, pos2 = (0, 0), (0, 0)
        player = 'e'
        board1 = board1.split('\n')
        board2 = board2.split('\n')
        for i in range(len(board1)):
            row1 = list(board1[i].split(' '))
            row2 = list(board2[i].split(' '))
            for j in range(len(row1)):
                if row1[j] != row2[j]:
                    if row2[j] == 'e':
                        pos1 = (i, j)
                        player = row1[j]
                    else:
                        pos2 = (i, j)
                        player = row2[j]

        return pos1, pos2, player


def main():
    if len(sys.argv) != 3:
        raise Exception("Must have two command line arguments")

    my_type = sys.argv[1]
    board_size = int(sys.argv[2])
    player = Player(player_type=my_type, board_size=board_size)

    while True:
        player.step()
        player.print_agent_move()


if __name__ == "__main__":
    main()

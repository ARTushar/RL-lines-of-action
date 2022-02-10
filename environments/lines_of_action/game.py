from collections import deque
from typing import List


pieceSquareTable8: list = [
    -80, -25, -20, -20, -20, -20, -25, -80,
    -25, 10, 10, 10, 10, 10, 10, -25,
    -20, 10, 25, 25, 25, 25, 10, -20,
    -20, 10, 25, 50, 50, 25, 10, -20,
    -20, 10, 25, 50, 50, 25, 10, -20,
    -20, 10, 25, 25, 25, 25, 10, -20,
    -25, 10, 10, 10, 10, 10, 10, -25,
    -80, -25, -20, -20, -20, -20, -25, -80
]

BOARD = List[List[int]]


class Game:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = self._create_board()
        self.first_player = -1  # -1 -> black
        self.second_player = 1  # 1 -> white , 0 -> empty
        self.current_player = self.first_player
        self.first_player_last_move = None
        self.second_player_last_move = None
        self.done = False
        self.winner = None

    def step(self, selected_pos, target_pos):
        is_valid_move = self.is_valid_move(self.board, selected_pos, target_pos, self.current_player)
        if not is_valid_move:
            self.done = True
            self.winner = self._get_opposition(self.current_player)
            return self.get_invalid_move_reward()

        if self.current_player == self.first_player:
            self.update_board(selected_pos, target_pos)
            reward = self.get_score(self.board, self.first_player, self.second_player)

            self.first_player_last_move = target_pos
            self.current_player = self.second_player
        else:
            self.update_board(selected_pos, target_pos)
            reward = self.get_score(self.board, self.second_player, self.first_player)

            self.second_player_last_move = target_pos
            self.current_player = self.first_player

        self.winner = self.get_winner()
        if self.winner is not None:
            self.done = True

        return reward

    def print_board(self):
        for row in self.board:
            for val in row:
                print(self.get_symbol(val), end=' ')
            print()

    @staticmethod
    def get_symbol(val):
        if val == -1:
            return 'B'
        if val == 1:
            return 'W'
        return 'E'

    @staticmethod
    def get_invalid_move_reward():
        return -10000

    @staticmethod
    def is_valid_move(board: BOARD, move_from, move_to, player_type):
        if board[move_from[0]][move_from[0]] != player_type:
            return False
        valid_moves = Game.get_valid_moves(board, move_from[0], move_from[1])
        for move in valid_moves:
            if move == move_to:
                return True
        return False

    def update_board(self, selected_pos, target_pos):
        player_type = self.board[selected_pos[0]][selected_pos[1]]
        self.board[selected_pos[0]][selected_pos[1]] = 0
        self.board[target_pos[0]][target_pos[1]] = player_type

    def reset_game(self):
        self.board = self._create_board()
        self.first_player = -1  # -1 -> black
        self.second_player = 1  # 1 -> white , 0 -> empty
        self.current_player = self.first_player
        self.first_player_last_move = None
        self.second_player_last_move = None
        self.done = False

    def _create_board(self):
        board: BOARD = []
        for i in range(self.board_size):
            board.append([])
            for j in range(self.board_size):
                if (i == 0 or i == self.board_size - 1) and j != 0 and j != self.board_size-1:
                    board[i].append(-1)
                elif (j == 0 or j == self.board_size - 1) and i != 0 and i != self.board_size-1:
                    board[i].append(1)
                else:
                    board[i].append(0)
        return board

    @staticmethod
    def _get_opposition(player):
        if player == -1:
            return 1
        elif player == 1:
            return -1

        return None

    @staticmethod
    def _get_total_active_cells(board: BOARD, row: int, col: int) -> tuple:
        row_count: int = 0
        col_count: int = 0
        left_diagonal_count: int = 0
        right_diagonal_count: int = 0

        for i in range(len(board)):
            col_count += abs(board[i][col])
            row_count += abs(board[row][i])

        i, j = row, col
        while i >= 0 and j >= 0:
            left_diagonal_count += abs(board[i][j])
            i -= 1
            j -= 1

        i, j = row+1, col+1
        while i < len(board) and j < len(board):
            left_diagonal_count += abs(board[i][j])
            i += 1
            j += 1

        i, j = row, col

        while i < len(board) and j >= 0:
            right_diagonal_count += abs(board[i][j])
            i += 1
            j -= 1

        i, j = row-1, col-1

        while i >= 0 and j < len(board):
            right_diagonal_count += abs(board[i][j])
            i -= 1
            j += 1

        return row_count, col_count, left_diagonal_count, right_diagonal_count

    @staticmethod
    def get_valid_moves(board, row, col):
        valid_moves = []
        player_type = board[row][col]
        opposition_type = Game._get_opposition(player_type)
        assert opposition_type is not None

        row_count, col_count, left_diagonal_count, right_diagonal_count = Game._get_total_active_cells(board, row, col)
        val_direction = {
            'left_row': True,
            'right_row': True,
            'up_col': True,
            'down_col': True,
            'left_up_diagonal': True,
            'right_down_diagonal': True,
            'right_up_diagonal': True,
            'left_down_diagonal': True
        }

        for i in range(1, row_count):
            if col - i < 0 or board[row][col - i] == opposition_type:
                val_direction['left_row'] = False

            if col + i >= len(board) or board[row][col] == opposition_type:
                val_direction['right_row'] = False

        if col-row_count < 0 or board[row][col-row_count] == player_type:
            val_direction['left_row'] = False
        if col+row_count >= len(board) or board[row][col+row_count] == player_type:
            val_direction['right_row'] = False

        for i in range(1, col_count):
            if row - i < 0 or board[row-i][col] == opposition_type:
                val_direction['up_col'] = False

            if row + i >= len(board) or board[row + i][col] == opposition_type:
                val_direction['down_col'] = False

        if row - col_count < 0 or board[row-col_count][col] == player_type:
            val_direction['up_col'] = False
        if row + col_count >= len(board) or board[row + col_count][col] == player_type:
            val_direction['down_col'] = False

        for i in range(1, left_diagonal_count):
            if row - i < 0 or col - i < 0 or board[row - i][col - i] == opposition_type:
                val_direction['left_up_diagonal'] = False

            if row + i >= len(board) or col + i >= len(board) or board[row + i][col + i] == opposition_type:
                val_direction['right_down_diagonal'] = False

        if row - left_diagonal_count < 0 or col - left_diagonal_count < 0 or board[row - left_diagonal_count][col - left_diagonal_count] == player_type:
            val_direction['left_up_diagonal'] = False
        if row + left_diagonal_count >= len(board) or col + left_diagonal_count >= len(board) or board[row + left_diagonal_count][col + left_diagonal_count] == player_type:
            val_direction['right_down_diagonal'] = False

        for i in range(1, right_diagonal_count):
            if row + i >= len(board) or col - i < 0 or board[row + i][col - i] == opposition_type:
                val_direction['left_down_diagonal'] = False

            if row - i < 0 or col + i >= len(board) or board[row - i][col + i] == opposition_type:
                val_direction['right_up_diagonal'] = False

        if row + right_diagonal_count >= len(board) or col - right_diagonal_count < 0 or board[row + right_diagonal_count][col - right_diagonal_count] == player_type:
            val_direction['left_down_diagonal'] = False
        if row - right_diagonal_count < 0 or col + right_diagonal_count >= len(board) or board[row - right_diagonal_count][col + right_diagonal_count] == player_type:
            val_direction['right_up_diagonal'] = False

        if val_direction['left_row']:
            valid_moves.append((row, col - row_count))
        if val_direction['right_row']:
            valid_moves.append((row, col + row_count))
        if val_direction['up_col']:
            valid_moves.append((row - col_count, col))
        if val_direction['down_col']:
            valid_moves.append((row + col_count, col))
        if val_direction['left_up_diagonal']:
            valid_moves.append((row - left_diagonal_count, col - left_diagonal_count))
        if val_direction['right_down_diagonal']:
            valid_moves.append((row + left_diagonal_count, col + left_diagonal_count))
        if val_direction['left_down_diagonal']:
            valid_moves.append((row + right_diagonal_count, col - right_diagonal_count))
        if val_direction['right_up_diagonal']:
            valid_moves.append((row - right_diagonal_count, col + right_diagonal_count))

        return valid_moves

    def get_all_current_player_moves(self):
        return self.get_all_moves(self.board, self.current_player)

    @staticmethod
    def get_all_moves(board: BOARD, player_type: int):
        all_valid_moves = []
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == player_type:
                    all_valid_moves.append(((i, j), Game.get_valid_moves(board, i, j)))

        return all_valid_moves

    @staticmethod
    def is_valid_position(board: BOARD, row: int, col: int) -> bool:
        board_len = len(board)
        return 0 <= row < board_len and 0 <= col < board_len

    @staticmethod
    def get_neighbors(board: BOARD, row: int, col: int) -> List[tuple]:
        neighbors: List[tuple] = []

        player_type = board[row][col]
        test_positions = [
            (row, col+1), (row, col-1), (row+1, col), (row-1, col),
            (row+1, col+1), (row+1, col-1), (row-1, col+1), (row-1, col-1)
        ]

        for test_row, test_col in test_positions:
            if Game.is_valid_position(board, row, col) and board[test_row][test_col] == player_type:
                neighbors.append((test_row, test_col))
        return neighbors

    @staticmethod
    def get_pos_index(board: BOARD, row: int, col: int) -> int:
        board_len = len(board)
        return board_len * row + col

    @staticmethod
    def bfs(board: BOARD, start_row: int, start_col: int) -> int:
        queue = deque()
        visited = set()
        start_pos = (start_row, start_col)
        queue.append(start_pos)
        visited.add(Game.get_pos_index(board, start_row, start_col))

        while len(queue) != 0:
            curr_row, curr_col = queue.popleft()
            for n_row, n_col in Game.get_neighbors(board, curr_row, curr_col):
                index = Game.get_pos_index(board, n_row, n_col)
                if index not in visited:
                    queue.append((n_row, n_col))
                    visited.add(Game.get_pos_index(board, n_row, n_col))

        return len(visited)

    @staticmethod
    def get_a_position(board: BOARD, player_type: int):
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == player_type:
                    return i, j
        return None

    @staticmethod
    def calculate_total_cells(board: BOARD, player_type):
        total_cells = 0
        for row in board:
            for val in row:
                if val == player_type:
                    total_cells += 1
        return total_cells

    def get_winner(self):
        board = self.board
        if self.current_player == self.first_player:
            player_last_move = self.first_player_last_move
            opposition_last_move = self.second_player_last_move
        else:
            player_last_move = self.first_player_last_move
            opposition_last_move = self.second_player_last_move

        if player_last_move is None or opposition_last_move is None:
            return None

        player_type = self.current_player
        opposition_type = Game._get_opposition(player_type)

        player_total_cells = Game.calculate_total_cells(board, player_type)
        opposition_total_cells = Game.calculate_total_cells(board, opposition_type)

        if opposition_total_cells == 0:
            return player_type

        if Game.bfs(board, player_last_move[0], player_last_move[1]) == player_total_cells:
            return player_type

        opposition_random_pos = None
        if player_last_move == opposition_last_move:
            opposition_random_pos = Game.get_a_position(board, opposition_type)

        if opposition_random_pos is None:
            return player_type

        if Game.bfs(board, opposition_random_pos[0], opposition_random_pos[1]) == opposition_total_cells:
            return opposition_type

        return None

    @staticmethod
    def get_score(board: BOARD, player_type, opposition_type):
        return Game.calculate_piece_square_sum(board, player_type, opposition_type)

    @staticmethod
    def calculate_piece_square_sum(board: BOARD, player_type: int, opposition_type: int) -> int:
        total_reward = 0
        board_len = len(board)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == player_type:
                    total_reward += pieceSquareTable8[i * board_len + j]
                elif board[i][j] == opposition_type:
                    total_reward -= pieceSquareTable8[i * board_len + j]
        return total_reward

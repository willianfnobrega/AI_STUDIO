import random


class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol):
        # This is the name that will show on the web page.
        self.name = "BasicRLAgent"

        # I save the symbols here so the agent knows which pieces are mine,
        # which spaces are empty, and which pieces belong to the opponent.
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol

        # This is my simple Q table.
        # It stores board positions and the score for moves from that position.
        self.q_table = {}

        # These are the learning settings.
        # learning_rate says how much the agent changes its old idea.
        # discount is here for reinforcement learning style, even though this
        # version only updates from the final result.
        # epsilon controls how often the agent tries a random move.
        self.learning_rate = 0.2
        self.discount = 0.9
        self.epsilon = 0.3
        self.games_played = 0

        # I remember the last board state and move so I can reward or punish it
        # when the game finishes.
        self.last_state = None
        self.last_action = None

    def _state_key(self, board):
        # Numpy arrays cannot be used directly as dictionary keys, so I flatten
        # the board and turn it into a tuple.
        return tuple(int(cell) for cell in board.flatten())

    def _valid_moves(self, board):
        size = board.shape[0]
        moves = set()

        # Instead of checking every empty space all the time, I first look for
        # empty spaces near pieces that are already on the board.
        for row in range(size):
            for col in range(size):
                if board[row, col] != self.blank_symbol:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            next_row, next_col = row + dr, col + dc
                            if 0 <= next_row < size and 0 <= next_col < size:
                                if board[next_row, next_col] == self.blank_symbol:
                                    moves.add((next_row, next_col))

        # If there are moves near existing pieces, use those.
        if moves:
            return list(moves)

        # If the board is empty, then there are no nearby pieces yet, so all
        # empty cells are possible moves.
        return [(row, col) for row in range(size) for col in range(size) if board[row, col] == self.blank_symbol]

    def _is_winning_move(self, board, move, symbol):
        size = board.shape[0]
        row, col = move

        # These are the four line directions: vertical, horizontal, and the two
        # diagonals.
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # I check both sides of the move because the line can continue in
            # either direction.
            for direction in (-1, 1):
                r, c = row, col
                for _ in range(4):
                    r += dr * direction
                    c += dc * direction
                    if 0 <= r < size and 0 <= c < size and board[r, c] == symbol:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False

    def _score_move(self, board, move):
        # Look up what the Q table thinks about this move.
        # If it never saw this board or move before, I just use 0.
        state_key = self._state_key(board)
        return self.q_table.get(state_key, {}).get(move, 0.0)

    def _choose_move(self, board, moves):
        size = board.shape[0]
        center = size // 2

        # This should almost never happen, but if there are no moves then I
        # return the center so the function still gives something back.
        if not moves:
            return (center, center)

        # Epsilon means exploration. Sometimes the agent picks a random move so
        # it can try new things instead of always doing the same move.
        if random.random() < self.epsilon:
            return random.choice(moves)

        # Otherwise I choose the move with the best Q-table score.
        best_move = None
        best_score = float("-inf")
        for move in moves:
            score = self._score_move(board, move)
            if score > best_score:
                best_score = score
                best_move = move
            elif score == best_score:
                if best_move is None:
                    best_move = move
                else:
                    # If two moves have the same score, I choose the one closer
                    # to the center because center moves are usually better in
                    # board games like Gomoku.
                    current_dist = abs(move[0] - center) + abs(move[1] - center)
                    best_dist = abs(best_move[0] - center) + abs(best_move[1] - center)
                    if current_dist < best_dist:
                        best_move = move

        # This is just a backup in case best_move was not set.
        if best_move is None:
            best_move = min(moves, key=lambda move: abs(move[0] - center) + abs(move[1] - center))

        return best_move

    def play(self, board):
        # On the first turn I play in the middle because that is a strong
        # starting place.
        if (board == self.blank_symbol).sum() == board.size:
            return (board.shape[0] // 2, board.shape[1] // 2)

        moves = self._valid_moves(board)

        # First I check if I can win immediately.
        for move in moves:
            board[move] = self.agent_symbol
            if self._is_winning_move(board, move, self.agent_symbol):
                board[move] = self.blank_symbol
                self.last_state = self._state_key(board)
                self.last_action = move
                return move
            board[move] = self.blank_symbol

        # If I cannot win, I check if the opponent could win next turn.
        # If yes, I block that move.
        for move in moves:
            board[move] = self.opponent_symbol
            if self._is_winning_move(board, move, self.opponent_symbol):
                board[move] = self.blank_symbol
                self.last_state = self._state_key(board)
                self.last_action = move
                return move
            board[move] = self.blank_symbol

        # If there is no instant win or block, I use the Q table and epsilon
        # logic to choose a move.
        move = self._choose_move(board, moves)
        self.last_state = self._state_key(board)
        self.last_action = move
        return move

    def on_game_over(self, board, winner):
        # At the end of the game I give a reward.
        # Win = good, lose = bad, draw = neutral.
        reward = 0.0
        if winner == self.agent_symbol:
            reward = 1.0
        elif winner is not None:
            reward = -1.0

        # Update the score for the last move the agent made.
        # This is a very basic Q-learning idea:
        # new value = old value + learning rate * difference
        if self.last_state is not None and self.last_action is not None:
            state_actions = self.q_table.setdefault(self.last_state, {})
            previous_value = state_actions.get(self.last_action, 0.0)
            state_actions[self.last_action] = previous_value + self.learning_rate * (reward - previous_value)

        # Count the game and slowly reduce epsilon so the agent explores less
        # after it has played more games.
        self.games_played += 1
        self.epsilon = max(0.05, self.epsilon * 0.995)

        # Clear these so the next game starts fresh.
        self.last_state = None
        self.last_action = None

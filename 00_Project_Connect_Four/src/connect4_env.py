import numpy as np

ROWS = 4
COLS = 4

class ConnectFourEnv:
    """
    Simple Connect Three environment (Tic-Tac-Toe style).
    Board: 4x4, values: 0 (empty), 1 (player 1), -1 (player 2)
    current_player: 1 or -1
    Players can place pieces anywhere on the board.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self._calculate_stats()

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.copy(), self.current_player

    def valid_actions(self):
        """Returns list of (row, col) tuples for empty positions."""
        return [(r, c) for r in range(ROWS) for c in range(COLS) if self.board[r, c] == 0]

    def _calculate_stats(self):
        """Calculate total possible winning combinations."""
        # Horizontal: 4 rows × 2 positions per row = 8
        horizontal = ROWS * (COLS - 2)
        
        # Vertical: 4 cols × 2 positions per col = 8
        vertical = COLS * (ROWS - 2)
        
        # Diagonal \ : 4 possible
        diagonal_down = (ROWS - 2) * (COLS - 2)
        
        # Diagonal / : 4 possible
        diagonal_up = (ROWS - 2) * (COLS - 2)
        
        self.total_winning_combinations = horizontal + vertical + diagonal_down + diagonal_up
        self.stats = {
            'horizontal': horizontal,
            'vertical': vertical,
            'diagonal_down': diagonal_down,
            'diagonal_up': diagonal_up,
            'total': self.total_winning_combinations
        }

    def step(self, action):
        """
        action: tuple (row, col) indicating position to place piece
        Returns: (next_state, reward, done, info)
        reward: 1.0 if current player wins, 0.5 draw, 0.0 otherwise
        """
        if isinstance(action, int):
            # Convert flat index to (row, col) for backward compatibility
            row = action // COLS
            col = action % COLS
            action = (row, col)
        
        row, col = action
        
        if action not in self.valid_actions():
            raise ValueError(f"Invalid action {action} for current board state.")

        self.board[row, col] = self.current_player

        if self._is_winner(self.current_player):
            reward = 1.0
            done = True
        elif len(self.valid_actions()) == 0:
            reward = 0.5
            done = True
        else:
            reward = 0.0
            done = False

        if not done:
            self.current_player *= -1

        return self.get_state(), reward, done, {}

    def _is_winner(self, player):
        b = self.board

        # Horizontal (Connect 3)
        for r in range(ROWS):
            for c in range(COLS - 2):
                if np.all(b[r, c:c+3] == player):
                    return True

        # Vertical (Connect 3)
        for r in range(ROWS - 2):
            for c in range(COLS):
                if np.all(b[r:r+3, c] == player):
                    return True

        # Diagonal / (Connect 3)
        for r in range(2, ROWS):
            for c in range(COLS - 2):
                if all(b[r - i, c + i] == player for i in range(3)):
                    return True

        # Diagonal \ (Connect 3)
        for r in range(ROWS - 2):
            for c in range(COLS - 2):
                if all(b[r + i, c + i] == player for i in range(3)):
                    return True

        return False

    def render(self):
        """Display the board with X and O symbols, legend, and stats."""
        print("\n" + "=" * 50)
        print("                  CONNECT 3 GAME")
        print("=" * 50)
        
        # Player legend
        symbol_map = {0: '·', 1: 'X', -1: 'O'}
        current_symbol = symbol_map[self.current_player]
        
        print("\n📋 LEGEND:")
        print("   X = Player 1 (You) 👤")
        print("   O = Player 2 (Opponent/AI) 🤖")
        print(f"   · = Empty space")
        print(f"\n▶ Current turn: {current_symbol} (Player {'1 (You)' if self.current_player == 1 else '2'})")
        
        # Game statistics
        print(f"\n📊 WINNING COMBINATIONS:")
        print(f"   Horizontal: {self.stats['horizontal']} possible")
        print(f"   Vertical: {self.stats['vertical']} possible")
        print(f"   Diagonal \\: {self.stats['diagonal_down']} possible")
        print(f"   Diagonal /: {self.stats['diagonal_up']} possible")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   TOTAL: {self.stats['total']} ways to win!")
        
        # Count pieces on board
        x_count = np.sum(self.board == 1)
        o_count = np.sum(self.board == -1)
        empty_count = np.sum(self.board == 0)
        
        print(f"\n🎯 BOARD STATUS:")
        print(f"   X pieces: {x_count}")
        print(f"   O pieces: {o_count}")
        print(f"   Empty: {empty_count}")
        
        # Print board
        print("\n🎮 BOARD:")
        print("     " + "   ".join(str(i) for i in range(COLS)) + "  (col)")
        print("   +" + "---+" * COLS)
        
        for r in range(ROWS):
            row_str = f" {r} | "
            row_str += " | ".join(symbol_map[int(cell)] for cell in self.board[r])
            row_str += " |"
            print(row_str)
            print("   +" + "---+" * COLS)
        
        print(" (row)")
        print("\n💡 TIP: Choose position as 'row col' (e.g., '0 1' for row 0, col 1)")
        print("=" * 50)

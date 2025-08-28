from enum import Enum

BOARD_SIZE = 3


class Turn(Enum):
    NOUGHTS = -1
    CROSSES = 1

    def __neg__(self) -> "Turn":
        """
        Returns the opposite player's turn.
        Returns:
            Turn: The opposite player's turn.
        """

        if self == Turn.NOUGHTS:
            return Turn.CROSSES
        return Turn.NOUGHTS


class NoughtsAndCrosses:
    def __init__(self) -> None:
        """
        Initializes a new game instance by creating an empty board, setting the starting player to NOUGHTS,
        and marking the game as running.
        """
        self._board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self._player = Turn.NOUGHTS
        self._game_running = True

    def make_move(self, row: int, col: int) -> None:
        """
        Makes a move for the current player at the specified row and column.
        This method checks if the move is valid, updates the board with the current player's symbol,
        checks for a win condition, prints the current board state, and switches turns if the game is still running.
        Args:
            row (int): The row index where the player wants to make a move.
            col (int): The column index where the player wants to make a move.
        Raises:
            ValueError: If the move is invalid (e.g., the cell is already occupied or out of bounds).
        """

        self._check_move(row, col)
        self._board[row][col] = self._player.value
        self._check_win(row, col)

        print(self)

        # switch turns
        if self._game_running:
            self._player = -self._player
            print(f"Your turn, {self._player.name}")

    @property
    def running(self) -> bool:
        """
        Check if the game is currently running.
        Returns:
            bool: True if the game is running, False otherwise.
        """

        return self._game_running

    def _check_move(self, row: int, col: int) -> None:
        """
        Validates a move by checking if the game is running, the specified row and column are within bounds,
        and the target tile is empty.
        Args:
            row (int): The row index of the move.
            col (int): The column index of the move.
        Raises:
            RuntimeError: If the game is not currently running.
            IndexError: If the row or column is out of the valid board range.
            ValueError: If the specified tile is already occupied.
        """

        if not self._game_running:
            msg = "Move not accepted, game is not running."
            raise RuntimeError(msg)

        if row >= BOARD_SIZE:
            msg = "Row out of range."
            raise IndexError(msg)
        if col >= BOARD_SIZE:
            msg = "Column out of range."
            raise IndexError(msg)

        if self._board[row][col] != 0:
            msg = f"Provided tile ({row}, {col}) is not empty."
            raise ValueError(msg)

    def _check_win(self, row: int, col: int) -> None:
        """
        Checks if the current move at the specified row and column results in a win or a draw.
        This method evaluates whether the player who made the most recent move has won the game.
        If a win is detected, it announces the winner and sets the game as not running.
        Args:
            row (int): The row index of the most recent move.
            col (int): The column index of the most recent move.
        """

        # get sums of row, col, diag
        row_sum = sum(t for t in self._board[row])
        col_sum = sum(self._board[r_idx][col] for r_idx in range(BOARD_SIZE))
        diag_sum = (
            sum(self._board[i][i] for i in range(BOARD_SIZE)) if row == col else 0
        )

        if (
            abs(row_sum) == BOARD_SIZE
            or abs(col_sum) == BOARD_SIZE
            or abs(diag_sum) == BOARD_SIZE
        ):
            print(f"{self._player.name} has won!")
            self._game_running = False

        # stalemate
        if all(t != 0 for row in self._board for t in row):
            print("All tiles are filled, the game ended in a draw!")
            self._game_running = False

    @property
    def turn(self) -> str:
        """
        Returns which player's turn it is.
        Returns:
            str: NOUGHTS/CROSSES.
        """

        return self._player.name

    def _format_tile(self, tile: int) -> str:
        if tile == -1:
            return "O"
        if tile == 1:
            return "X"
        if tile == 0:
            return " "
        msg = "Invalid tile data provided."
        raise ValueError(msg)

    def __str__(self) -> str:
        flat_tiles = [self._format_tile(tile) for row in self._board for tile in row]
        return "\
    {}|{}|{}\n\
    -----\n\
    {}|{}|{}\n\
    -----\n\
    {}|{}|{}".format(*flat_tiles)


# Further Suggestions
# Add agent class? same idea as rock paper scissors but more complex
# Refactor to make BOARD_SIZE an input at construction


if __name__ == "__main__":
    from random import randint

    game = NoughtsAndCrosses()
    while game.running:
        move = (randint(0, 2), randint(0, 2))
        try:
            game.make_move(*move)
        except ValueError:
            print(f"Move {move} not valid in: {game}\n")

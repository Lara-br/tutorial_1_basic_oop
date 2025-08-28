from enum import Enum

BOARD_SIZE = 3

class Turn(Enum):
    NOUGHTS = -1
    CROSSES = 1
    
    def __neg__(self) -> "Turn":
        if self == Turn.NOUGHTS:
            return Turn.CROSSES
        else:
            return Turn.NOUGHTS

class NoughtsAndCrosses:
    def __init__(self):
        # make board
        self._board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self._player = Turn.NOUGHTS
        self._game_running = True
        
    def make_move(self, row: int, col: int):
        # make move for player in turn
        self._check_move(row, col) 
        self._board[row][col] = self._player.value
        self._check_win(row, col)
        
        print(self)
        
        # switch turns
        if self._game_running:
            self._player = -self._player
            print(f"Your turn, {self._player.name}")
        
    @property
    def running(self):
        return self._game_running
       
    def _check_move(self, row: int, col: int) -> None:
        if not self._game_running:
            raise RuntimeError("Move not accepted, game is not running.")
        
        if row >= BOARD_SIZE: # Can also be made a constant
            raise IndexError("Row out of range.")
        if col >= BOARD_SIZE: # Can also be made a constant
            raise IndexError("Column out of range.")

        if self._board[row][col] != 0:
            raise ValueError(f"Provided tile ({row}, {col}) is not empty.")

    def _check_win(self, row: int, col: int):
        # check whether a win occured by making given square x or o
        row_sum = sum(t for t in self._board[row])
        col_sum = sum(self._board[r_idx][col] for r_idx in range(BOARD_SIZE))
        diag_sum = sum(self._board[i][i] for i in range(BOARD_SIZE)) if row == col else 0
        
        if abs(row_sum) == BOARD_SIZE or abs(col_sum) == BOARD_SIZE or abs(diag_sum) == BOARD_SIZE:
            print(f"{self._player.name} has won!")
            self._game_running = False
        
        # stalemate
        
        if all(t != 0 for row in self._board for t in row):
            print(f"All tiles are filled, the game ended in a draw!")
            self._game_running = False 
            
    @property
    def turn(self):
        return self._player.name

    def _format_tile(self, tile: int) -> str:
        if tile == -1:
            return "O"
        elif tile == 1:
            return "X"
        elif tile == 0:
            return " "
        else:
            raise ValueError("Invalid tile data provided.")
    
    def __str__(self):
        flat_tiles = [self._format_tile(tile) for row in self._board for tile in row]
        return ("\
    {}|{}|{}\n\
    -----\n\
    {}|{}|{}\n\
    -----\n\
    {}|{}|{}".format(*flat_tiles))

# Further Suggestions
# Add agent class? same idea as rock paper scissors but more complex
# Refactor to make BOARD_SIZE an input at construction


if __name__ == "__main__":
    from random import randint
    game = NoughtsAndCrosses()
    while game.running:
        move = (randint(0,2), randint(0,2))
        try:
            game.make_move(*move)
        except ValueError:
            print(f"Move {move} not valid in: {game}\n")
from abc import ABC, abstractmethod
from enum import Enum
from random import choices


class Move(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    def beats(self, other: "Move") -> bool:
        """
        Determines if this move beats another move in the game of Rock-Paper-Scissors.
        Args:
            other (Move): The move to compare against.
        Returns:
            bool: True if this move beats the other move, False otherwise.
        """

        if self == Move.ROCK and other == Move.SCISSORS:
            return True
        if self == Move.PAPER and other == Move.ROCK:
            return True
        return bool(self == Move.SCISSORS and other == Move.PAPER)


class GameAgent(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def move(*args) -> Move:
        pass


class LearningAgent(GameAgent, ABC):
    @abstractmethod
    def update(self, result) -> None:
        pass


class WeightedAgent(GameAgent):
    def __init__(self, prob: list = [1 / len(Move)] * len(Move)) -> None:
        """
        Initializes the object with a probability distribution for moves.
        Args:
            prob (list, optional): A list of probabilities corresponding to each move in `Move`.
                Defaults to a uniform distribution over all moves.
        Raises:
            ValueError: If the provided probability list is invalid (e.g., wrong length, negative values, or does not sum to 1).
        """

        self.validate_prob(prob)
        self._prob = prob

    @staticmethod
    def validate_prob(prob) -> None:
        """
        Validates that the given probability distribution is valid for the set of moves.
        Args:
            prob (list or sequence of float): Probabilities assigned to each move.
        Raises:
            ValueError: If the probabilities do not sum to 1.
            ValueError: If the number of probabilities does not match the number of moves.
        """

        if sum(prob) != 1:
            msg = f"prob {prob} does not sum to 1."
            raise ValueError(msg)
        if len(prob) != len(Move):
            msg = f"Each move must be assigned a probability in {prob}"
            raise ValueError(msg)

    def move(self) -> Move:
        """
        Selects and returns a move based on predefined probabilities.
        Returns:
            Move: The selected move, chosen randomly according to the weights specified in self._prob.
        """

        return choices(list(Move), weights=self._prob)[0]


class AverageRewardAgent(WeightedAgent, LearningAgent):
    def __init__(self, prob=[1 / len(Move)] * len(Move)) -> None:
        """
        Initializes the object with a probability distribution over moves.
        Args:
            prob (list, optional): A list of probabilities for each move. Defaults to a uniform distribution over all moves.
        """

        super().__init__(prob)
        self._rewards = [0] * len(Move)
        self._total_reward = 0

    def update(self, move: Move, win: bool) -> None:
        """
        Updates the internal reward and probability distributions based on the outcome of a move.
        Args:
            move (Move): The move that was played.
            win (bool): Whether the move resulted in a win (True) or not (False).
        """

        self._rewards[move.value] += int(win)
        self._total_reward += int(win)

        raw = [self._rewards[i] + 1 for i in range(len(Move))]
        s = sum(raw)
        self._prob = [x / s for x in raw]


class RockPaperScissors:
    # Alternatively make this more fleshed out adding diff methods like hist etc.
    @staticmethod
    def run(agent_1: GameAgent, agent_2: GameAgent, verbose: bool = True) -> bool:
        """
        Executes a game round between two GameAgent instances and determines the winner.
        Args:
            agent_1 (GameAgent): The first agent participating in the game.
            agent_2 (GameAgent): The second agent participating in the game.
            verbose (bool, optional): If True, prints the result of the game round. Defaults to True.
        Returns:
            bool: True if agent_1 wins, False if agent_2 wins.
        """

        move_1 = agent_1.move()
        move_2 = agent_2.move()
        result = move_1.beats(move_2)
        if verbose:
            print(
                f"Agent {'1' if result else '2'} wins because {move_1 if result else move_2} beats {move_2 if result else move_1}",
            )
        return result

    @staticmethod
    def train(learner: LearningAgent, teacher: GameAgent) -> bool:
        """
        Trains a LearningAgent by having it play a single round against a teacher GameAgent.
        Args:
            learner (LearningAgent): The agent that is learning and will be updated based on the outcome.
            teacher (GameAgent): The agent acting as the teacher, providing moves for the learner to play against.
        Returns:
            bool: The result of the round, typically indicating if the learner's move beats the teacher's move.
        """

        lm = learner.move()
        tm = teacher.move()
        result = lm.beats(tm)
        learner.update(lm, result)
        return result


if __name__ == "__main__":
    bot1 = WeightedAgent([1 / 10, 1 / 10, 8 / 10])
    hist = {Move.ROCK: 0, Move.PAPER: 0, Move.SCISSORS: 0}
    for _ in range(1000):
        m = bot1.move()
        hist[m] += 1
    print(hist)

    bot2 = WeightedAgent([1, 0, 0])
    game = RockPaperScissors()

    for _ in range(10):
        game.run(bot1, bot2)

    learning_bot = AverageRewardAgent()

    for _ in range(1000):
        game.train(learning_bot, bot1)

    print(learning_bot._prob)
    wins = 0
    for _ in range(1000):
        wins += game.run(learning_bot, bot1, False)
    print(wins)

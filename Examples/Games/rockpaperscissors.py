from enum import Enum
from random import choices, random
from abc import ABC, abstractmethod
# Currently purposefully without numpy, can include if better

class Move(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    
    def beats(self, other: 'Move') -> bool:
        if self == Move.ROCK and other == Move.SCISSORS:
            return True
        if self == Move.PAPER and other == Move.ROCK:
            return True
        if self == Move.SCISSORS and other == Move.PAPER:
            return True
        return False
    
class GameAgent(ABC):
    # Consider adding an update method to implement some basic reinforcement learning aspects
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def move(*args) -> Move:
        pass

class LearningAgent(GameAgent, ABC):
    @abstractmethod
    def update(selected_move, result) -> None:
        pass

class WeightedAgent(GameAgent):
    def __init__(self, prob: list = [1/len(Move)] * len(Move)):
        self.validate_prob(prob)
        self._prob = prob
    
    @staticmethod
    def validate_prob(prob):
        if sum(prob) != 1:
            raise ValueError(f'prob {prob} does not sum to 1.')
        if len(prob) != len(Move):
            raise ValueError(f'Each move must be assigned a probability in {prob}')
        
    def move(self) -> Move:
        return choices(list(Move), weights=self._prob)[0]
    
class AverageRewardAgent(WeightedAgent, LearningAgent):
    def __init__(self, prob = [1/len(Move)] * len(Move)):
        super().__init__(prob)
        self._rewards = [0] * len(Move)
        self._total_reward = 0
    
    def update(self, move: Move, win: bool) -> None:
        self._rewards[move.value] += int(win)
        self._total_reward += int(win)

        raw = [self._rewards[i] + 1 for i in range(len(Move))]
        s = sum(raw)
        self._prob = [x / s for x in raw]
        

class RockPaperScissors:
    # Alternatively make this more fleshed out adding diff methods like hist etc.
    @staticmethod
    def run(agent_1: GameAgent, agent_2: GameAgent, verbose: bool = True) -> bool:
        move_1 = agent_1.move()
        move_2 = agent_2.move()
        result = move_1.beats(move_2)
        if verbose:
            print(f"Agent {'1' if result else '2'} wins because {move_1 if result else move_2} beats {move_2 if result else move_1}")
        return result
    
    @staticmethod
    def train(learner: LearningAgent, teacher: GameAgent) -> bool:
        lm = learner.move()
        tm = teacher.move()
        result = lm.beats(tm)
        learner.update(lm, result)
        
    
if __name__ == "__main__":
    bot1 = WeightedAgent([1/10, 1/10, 8/10])
    hist = {Move.ROCK : 0, Move.PAPER : 0, Move.SCISSORS: 0}
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
        
        
    
    
import numpy as np
from typing import Tuple, List
from battleship.ai.base_ai import BattleshipAI
from battleship.plate_state_processor import WellState
import random

class RandomAI(BattleshipAI):
    """
    A random AI implementation for Battleship.
    This can be used as an example for students or as a default competitor.
    """
    def select_next_move(self) -> Tuple[int, int]:
        """
        Args:
            board: 2-D list of ints representing current knowledge of the enemy grid.

        Returns:
            (row, col) of the chosen target.  If the board has
            no UNKNOWN cells left, returns (0, 0) just like the Go version.
        """
        unknowns: List[Tuple[int, int]] = [
            (r, c)
            for r, row in enumerate(self.board_state)
            for c, cell in enumerate(row)
            if cell == WellState.UNKNOWN
        ]

        if not unknowns:
            return 0, 0

        return random.choice(unknowns)
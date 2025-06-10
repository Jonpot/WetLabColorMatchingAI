from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import numpy as np
from battleship.plate_state_processor import WellState

class BattleshipAI(ABC):
    """
    Abstract Base Class for a Battleship AI.
    Students will inherit from this class to create their own AI implementation.
    """

    def __init__(self, player_id: str, board_shape: Tuple[int, int], ship_schema: Dict[str, Any]):
        """
        Initializes the AI.

        Parameters
        ----------
        player_id : str
            A unique identifier for the player (e.g., 'player_1').
        board_shape : Tuple[int, int]
            The dimensions of the game board (rows, columns).
        ship_schema : Dict[str, Any]
            A dictionary describing the ships to be sunk (lengths and counts).
        """
        self.player_id = player_id
        self.board_shape = board_shape
        self.ship_schema = ship_schema
        self.board_state = np.full(board_shape, WellState.UNKNOWN, dtype=WellState)

    @abstractmethod
    def select_next_move(self) -> Tuple[int, int]:
        """
        Determine the next well to target. This is the core method students must implement.

        The method should return a tuple of (row, column) for the next shot.
        The AI should not target wells that have already been fired upon.

        Returns
        -------
        Tuple[int, int]
            The (row, column) coordinates for the next missile strike.
        """
        pass

    def record_shot_result(self, move: Tuple[int, int], result: WellState) -> None:
        """
        Updates the AI's internal board state with the result of a shot.

        Parameters
        ----------
        move : Tuple[int, int]
            The (row, column) of the shot.
        result : WellState
            The result of the shot (HIT or MISS).
        """
        row, col = move
        if self.board_state[row, col] == WellState.UNKNOWN:
            self.board_state[row, col] = result
        else:
            print(f"Warning ({self.player_id}): Attempted to record a result for an already targeted well {move}.")

    def has_won(self) -> bool:
        """
        Checks if the AI has won the game.

        Returns
        -------
        bool
            True if all opponent ships are sunk, False otherwise.
        """
        total_ship_segments = sum(ship['length'] * ship['count'] for ship in self.ship_schema.values())
        current_hits = np.sum(self.board_state == WellState.HIT)
        return current_hits >= total_ship_segments
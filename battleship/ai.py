"""
AI for Battleship Game

Will take as input the board state and return the best next move.
"""


from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Tuple
from string import ascii_uppercase
import numpy as np

from battleship.plate_state_processor import PlateStateProcessor, WellState
from battleship.robot.ot2_utils import OT2Manager


class AI:
    """AI for the Battleship game that determines the next move based on the plate state."""

    def __init__(self, plate_processor: PlateStateProcessor, robot: OT2Manager, ship_schema: Dict[str, Any]) -> None:
        """Initialize the AI with a plate schema and camera index."""
        self.plate_processor = plate_processor
        self.robot = robot
        self.ship_schema = ship_schema

        num_rows = plate_processor.plate_schema['rows']
        num_columns = plate_processor.plate_schema['columns']
        if not isinstance(num_rows, int) or not isinstance(num_columns, int):
            raise ValueError("Plate schema must contain 'rows' and 'columns' as integers.")

        self.board_state: np.ndarray = np.zeros((num_rows, num_columns), dtype=WellState)
        self.board_history: List[np.ndarray] = []
        self.board_history.append(self.board_state.copy())

    def main_loop(self) -> None:
        """Main loop for the AI to run until the game is over."""
        num_steps = 0
        while not self.determine_game_state():
            num_steps += 1
            next_move = self.get_next_move()
            if next_move:
                self.fire_missile(next_move)
                self.board_history.append(self.board_state.copy())
                print(f"Step {num_steps}: Fired at {ascii_uppercase[next_move[0]]}{next_move[1] + 1}.")
                print(f"It was a {self.board_state[next_move[0], next_move[1]]}!")
            else:
                print("No valid moves available. Exiting.")
                break
        print(f"Game Over! All ships sunk after {num_steps} steps.")

    def get_next_move(self) -> Tuple[int, int]:
        """Determine the next move based on the current board state."""
        raise NotImplementedError

    def fire_missile(self, targeted_well: Tuple[int, int]) -> None:
        """Fire a missile at the targeted well and updates the board state."""
        well_name = f"{ascii_uppercase[targeted_well[0]]}{targeted_well[1] + 1}"
        self.robot.add_fire_missile_action(well_name)
        self.robot.execute_actions_on_remote()

        well_state = self.plate_processor.determine_well_state(targeted_well[0], targeted_well[1])
        self.board_state[targeted_well[0], targeted_well[1]] = well_state

    def determine_game_state(self) -> bool:
        """Determine the current game state based on the board state.

        This should consider all the ships and their statuses based on the ship schema
        and the current board state.

        If all ships are sunk, the game is over.
        If any ship is still afloat, the game continues.

        This can be simplified to checking if there are as many hits as there are total ship segments.

        Returns:
            bool: True if the game is over (all ships sunk), False otherwise.
        """
        total_segments = sum(ship['length'] * ship['count'] for ship in self.ship_schema.values())
        hit_segments = np.sum(self.board_state == WellState.HIT)

        return hit_segments >= total_segments
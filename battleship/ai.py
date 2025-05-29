"""
AI for Battleship Game

Will take as input the board state and return the best next move.
"""


from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, Tuple

from battleship.plate_state_processor import PlateStateProcessor, WellState


class AI:
    """AI for the Battleship game that determines the next move based on the plate state."""
    
    def __init__(self, plate_schema: Dict[str, Any], cam_index: int = 2) -> None:
        """Initialize the AI with a plate schema and camera index."""
        self.plate_processor = PlateStateProcessor(plate_schema, cam_index)

    def get_next_move(self, targeted_well: Tuple[int, int]) -> WellState:
        """Get the state of the targeted well."""
        raise NotImplementedError
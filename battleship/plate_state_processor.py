from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera.camera_w_calibration import PlateProcessor
from enum import Enum
from typing import List, Dict, Any, Tuple
import numpy as np

class WellState(Enum):
    UNKNOWN = 0
    MISS = 1
    HIT = 2

class WellColor(Enum):
    CLEAR = 0
    RED = 1

class PlateStateProcessor:
    """A class to process the state of a plate based on camera input.
    
    Primary functionality is to determine the state of a well in a plate
    as a HIT or MISS based on the color detected in the well's position.
    """
    def __init__(self, plate_schema: Dict[str, Any], cam_index: int = 2) -> None:
        """Initialize the PlateStateProcessor with a camera index."""
        self.cam_index = cam_index
        self.processor = PlateProcessor()
        self.plate_schema = plate_schema

    def determine_well_state(self, well: Tuple[int, int]) -> WellState:
        """Determine the state of a well based on its coordinates.
        
        This will either be MISS or HIT. We assume that if the AI is
        asking for this well to be resolved, it has already been
        targeted by the AI and thus the state is known.
        """
        i, j = well

        rows = int(self.plate_schema.get('rows', 0))
        cols = int(self.plate_schema.get('columns', 0))
        if i < 0 or i >= rows or j < 0 or j >= cols:
            raise ValueError(f"Invalid well coordinates: {well}")

        plate_state = self.process_plate()

        well_color = plate_state[i][j]
        if well_color == WellColor.RED:
            return WellState.HIT
        elif well_color == WellColor.CLEAR:
            return WellState.MISS
        else:
            raise ValueError(f"Unknown well color: {well_color} at coordinates {well}")

    def process_plate(self) -> np.ndarray[WellColor]:
        """Process the plate image and return the measured plate as a WellColor array."""
        raw_plate = self.processor.process_image(cam_index=self.cam_index)
        return np.array([[self.determine_color(color) for color in row] for row in raw_plate])

    def determine_color(self, color: Tuple[int, int, int]) -> WellColor:
        """Determine the WellColor of a given pixel.
        
        Colors are determined based on RGB values. Because we are working with
        subtractive colors, simply having a high red value is not enough to
        determine if the well is red. We need to check if the red value is
        significantly higher than the other two color channels.
        """
        r, g, b = color
        if r > g + 30 and r > b + 30:
            return WellColor.RED
        else:
            return WellColor.CLEAR

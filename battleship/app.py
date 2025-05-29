from pathlib import Path
import sys
from typing import Any, Dict 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera.camera_w_calibration import PlateProcessor
from enum import Enum

import numpy as np

class WellState(Enum):
    UNKNOWN = 0
    MISS = 1
    HIT = 2

class WellColor(Enum):
    CLEAR = 0
    RED = 1

class PlateStateProcessor:
    def __init__(self, cam_index: int = 2, plate_schema: Dict[str, Any]) -> None:
        """Initialize the PlateStateProcessor with a camera index."""
        self.cam_index = cam_index
        self.processor = PlateProcessor()
        self.plate_schema = plate_schema

    def determine_well_state(self, well: tuple[int, int]) -> WellState:

    def process_plate(self) -> np.ndarray[WellColor]:
        """Process the plate image and return the measured plate as a WellColor array."""
        raw_plate = self.processor.process_image(cam_index=self.cam_index)
        return np.array([[self.determine_color(color) for color in row] for row in raw_plate])

    def determine_color(self, color: tuple[int, int, int]) -> WellColor:
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

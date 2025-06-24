from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera.camera_w_calibration import PlateProcessor
from camera.dual_camera_w_calibration import DualPlateProcessor
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np


def calibration_colors(plate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (miss_avg, hit_avg) using column 12 of the plate."""
    col = plate[:, 11]
    miss_avg = col[:4].mean(axis=0)
    hit_avg = col[4:8].mean(axis=0)
    return miss_avg, hit_avg

class WellState(Enum):
    UNKNOWN = 0
    MISS = 1
    HIT = 2

class PlateStateProcessor:
    """A class to process the state of a plate based on camera input.
    
    Primary functionality is to determine the state of a well in a plate
    as a HIT or MISS based on the color detected in the well's position.
    """
    def __init__(self, plate_schema: Dict[str, Any], ot_number: int = 2, cam_index: int = 2, virtual_mode: bool = False) -> None:
        """Initialize the PlateStateProcessor with a camera index."""
        self.cam_index = cam_index
        self.processor = PlateProcessor(virtual_mode=virtual_mode)
        self.plate_schema = plate_schema
        self.ot_number = ot_number

    def determine_well_state(self, well: Tuple[int, int]) -> WellState:
        """Determine the state of a well based on its coordinates using calibration wells."""
        i, j = well

        rows = int(self.plate_schema.get('rows', 0))
        cols = int(self.plate_schema.get('columns', 0))
        if i < 0 or i >= rows or j < 0 or j >= cols:
            raise ValueError(f"Invalid well coordinates: {well}")

        plate_colors = self.process_plate()
        miss_avg, hit_avg = calibration_colors(plate_colors)

        color = plate_colors[i, j]
        dist_miss = np.linalg.norm(color - miss_avg)
        dist_hit = np.linalg.norm(color - hit_avg)
        return WellState.MISS if dist_miss < dist_hit else WellState.HIT

    def process_plate(self) -> np.ndarray:
        """Return the measured plate colors."""
        return self.processor.process_image(
            cam_index=self.cam_index,
            calib=f"secret/OT_{self.ot_number}/calibration.json",
        )


class DualPlateStateProcessor:
    """A class to process the state of two plates based on camera input.
    
    Primary functionality is to determine the state of a well in a plate
    as a HIT or MISS based on the color detected in the well's position.
    """
    def __init__(self, plate_schema: Dict[str, Any], ot_number: int = 2, cam_index: int = 2, virtual_mode: bool = False) -> None:
        """Initialize the PlateStateProcessor with a camera index."""
        self.cam_index = cam_index
        self.processor = DualPlateProcessor(virtual_mode=virtual_mode)
        self.plate_schema = plate_schema
        self.ot_number = ot_number

    def determine_well_state(self, plate_id: int, well: Tuple[int, int]) -> WellState:
        """Determine the state of a well using calibration wells."""
        i, j = well

        rows = int(self.plate_schema.get("rows", 0))
        cols = int(self.plate_schema.get("columns", 0))
        if i < 0 or i >= rows or j < 0 or j >= cols:
            raise ValueError(f"Invalid well coordinates: {well}")

        plate_colors = self.process_plate(plate_id=plate_id)
        miss_avg, hit_avg = calibration_colors(plate_colors)

        color = plate_colors[i, j]
        dist_miss = np.linalg.norm(color - miss_avg)
        dist_hit = np.linalg.norm(color - hit_avg)
        return WellState.MISS if dist_miss < dist_hit else WellState.HIT

    def process_plate(self, plate_id: int) -> np.ndarray:
        """Return the measured plate colors for a given plate."""
        raw_plates = self.processor.process_image(
            cam_index=self.cam_index,
            calib=f"secret/OT_{self.ot_number}/dual_calibration.json",
        )
        raw_plate = raw_plates[f"plate_{plate_id}"]
        if raw_plate is None:
            raise ValueError(f"No plate data found for plate ID {plate_id}")
        return raw_plate
    

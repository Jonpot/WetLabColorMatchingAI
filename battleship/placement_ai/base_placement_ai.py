from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class PlacementAI(ABC):
    """Base class for ship placement algorithms."""

    def __init__(self, board_shape: Tuple[int, int], ship_schema: Dict[str, Any]):
        self.board_shape = board_shape
        self.ship_schema = ship_schema

    @abstractmethod
    def generate_placement(self) -> List[Dict[str, Any]]:
        """Return a placement schema for all ships."""
        raise NotImplementedError

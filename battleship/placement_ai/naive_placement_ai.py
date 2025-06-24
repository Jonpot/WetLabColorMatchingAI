from typing import Dict, Any, Tuple, List
from .base_placement_ai import PlacementAI

class NaivePlacementAI(PlacementAI):
    """Simple placement algorithm that packs ships row by row."""

    def generate_placement(self) -> List[Dict[str, Any]]:
        rows, cols = self.board_shape
        placements: List[Dict[str, Any]] = []
        current_row = 0
        current_col = 0
        for ship in self.ship_schema.values():
            length = ship['length']
            for _ in range(ship['count']):
                if current_col + length > cols:
                    current_row += 1
                    current_col = 0
                placements.append({
                    'row': current_row,
                    'col': current_col,
                    'length': length,
                    'direction': 'horizontal'
                })
                current_col += length + 1
        return placements

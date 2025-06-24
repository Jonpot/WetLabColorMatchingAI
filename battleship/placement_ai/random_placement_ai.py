import random
from typing import Dict, Any, Tuple, List
from .base_placement_ai import PlacementAI

class RandomPlacementAI(PlacementAI):
    """Random placement algorithm with basic collision avoidance."""

    def generate_placement(self) -> List[Dict[str, Any]]:
        rows, cols = self.board_shape
        taken = set()
        placements: List[Dict[str, Any]] = []
        for ship in self.ship_schema.values():
            length = ship['length']
            for _ in range(ship['count']):
                placed = False
                for _ in range(100):
                    direction = random.choice(['horizontal', 'vertical'])
                    if direction == 'horizontal':
                        row = random.randint(0, rows - 1)
                        col = random.randint(0, cols - length)
                        cells = {(row, col + i) for i in range(length)}
                    else:
                        row = random.randint(0, rows - length)
                        col = random.randint(0, cols - 1)
                        cells = {(row + i, col) for i in range(length)}
                    if not (cells & taken):
                        taken.update(cells)
                        placements.append({'row': row, 'col': col, 'length': length, 'direction': direction})
                        placed = True
                        break
                if not placed:
                    raise RuntimeError('Failed to place ship')
        return placements

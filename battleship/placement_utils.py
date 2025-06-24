from typing import List, Dict, Tuple, Any


def validate_placement_schema(schema: List[Dict[str, Any]], board_shape: Tuple[int, int], ship_schema: Dict[str, Any]) -> bool:
    """Validate placement schema for overlaps and bounds."""
    rows, cols = board_shape
    occupied = set()
    count = 0
    for entry in schema:
        row = entry['row']
        col = entry['col']
        length = entry['length']
        direction = entry['direction']
        if direction not in ('horizontal', 'vertical'):
            return False
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
        if direction == 'horizontal':
            if col + length > cols:
                return False
            cells = {(row, col + i) for i in range(length)}
        else:
            if row + length > rows:
                return False
            cells = {(row + i, col) for i in range(length)}
        if occupied & cells:
            return False
        occupied.update(cells)
        count += length
    expected = sum(s['length'] * s['count'] for s in ship_schema.values())
    if count != expected:
        return False
    return True

def coords_from_schema(schema: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for entry in schema:
        r = entry['row']
        c = entry['col']
        l = entry['length']
        d = entry['direction']
        if d == 'horizontal':
            coords.extend([(r, c + i) for i in range(l)])
        else:
            coords.extend([(r + i, c) for i in range(l)])
    return coords

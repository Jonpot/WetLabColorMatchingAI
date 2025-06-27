from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

ROWS = [chr(i) for i in range(ord("A"), ord("H") + 1)]
COLS = list(range(1, 13))
DATA_PATH = Path(__file__).resolve().parent / "well_data.json"
GLOBAL_DATA_PATH = Path(__file__).resolve().parent / "global_well_data.json"

def empty_table() -> Dict[str, Dict[str, List[int] | str]]:
    """Return a blank well table."""
    table: Dict[str, Dict[str, List[int] | str]] = {}
    for r in ROWS:
        for c in COLS:
            key = f"{r}{c}"
            recipe: List[int] | str
            if c == 1:
                recipe = "unknown"
            else:
                recipe = "empty"
            table[key] = {"recipe": recipe, "rgb": [0, 0, 0]}
    return table

def _load(path: Path) -> Dict[str, Dict[str, List[int] | str]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return empty_table()


def load_table() -> Dict[str, Dict[str, List[int] | str]]:
    return _load(DATA_PATH)


def load_global_table() -> Dict[str, Dict[str, List[int] | str]]:
    return _load(GLOBAL_DATA_PATH)

def _save(path: Path, table: Dict[str, Dict[str, List[int] | str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)


def save_table(table: Dict[str, Dict[str, List[int] | str]]) -> None:
    _save(DATA_PATH, table)


def save_global_table(table: Dict[str, Dict[str, List[int] | str]]) -> None:
    _save(GLOBAL_DATA_PATH, table)


def clear_saved_tables() -> Dict[str, Dict[str, List[int] | str]]:
    """
    Clears both the local and global saved tables.
    """
    table = empty_table()
    save_table(table)
    save_global_table(table)
    return table

def clear_current_saved_table() ->  Dict[str, Dict[str, List[int] | str]]:
    """
    Clears only the local table, not the global table.
    """
    table = empty_table()
    save_table(table)
    return table

def restore_global_table() ->  Dict[str, Dict[str, List[int] | str]]:
    """
    Restores the local table from the global table.
    """
    global_table = load_global_table()
    save_table(global_table)
    return global_table

def update_rgb_values(table: Dict[str, Dict[str, List[int] | str]], full_plate: List[List[List[int]]]) -> None:
    for r, row_letter in enumerate(ROWS):
        for c, col in enumerate(COLS):
            key = f"{row_letter}{col}"
            table[key]["rgb"] = [int(v) for v in full_plate[r][c]]

def set_well_recipe(table: Dict[str, Dict[str, List[int] | str]], well: str, recipe: List[int]) -> None:
    table[well]["recipe"] = recipe

def record_measurements(
    well_data: Dict[str, Dict[str, List[int] | str]],
    global_data: Dict[str, Dict[str, List[int] | str]],
    full_plate: List[List[List[int]]],
) -> None:
    """Update RGB values in both tables and persist them."""
    update_rgb_values(well_data, full_plate)
    update_rgb_values(global_data, full_plate)
    save_table(well_data)
    save_global_table(global_data)


def record_recipe(
    well_data: Dict[str, Dict[str, List[int] | str]],
    global_data: Dict[str, Dict[str, List[int] | str]],
    well: str,
    recipe: List[int],
) -> None:
    """Store recipe in both tables and persist them."""
    set_well_recipe(well_data, well, recipe)
    set_well_recipe(global_data, well, recipe)
    save_table(well_data)
    save_global_table(global_data)

def populate_optimizer(
    table: Dict[str, Dict[str, List[int] | str]],
    optimizer,
) -> None:
    """Load recipe/RGB data into the optimizer."""
    optimizer.X_train = []
    optimizer.Y_train = []
    for entry in table.values():
        recipe = entry["recipe"]
        rgb = entry["rgb"]
        if isinstance(recipe, list):
            optimizer.X_train.append(recipe)
            optimizer.Y_train.append(rgb)

    # And retrain the optimizer
    optimizer.train()

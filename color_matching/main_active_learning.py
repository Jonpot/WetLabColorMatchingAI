"""Utility functions for running the colour learning pipeline.

This module exposes ``run_active_learning`` which can be imported by
external applications.  When executed directly it behaves like the old
script and performs colour matching over a set of rows.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Callable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from color_matching.robot.ot2_utils import OT2Manager, TiprackEmptyError
from color_matching.active_learning.color_learning import ColorLearningOptimizer
from camera.camera_w_calibration import PlateProcessor


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def active_learn_row(
    robot: OT2Manager,
    processor: PlateProcessor,
    optimizer: ColorLearningOptimizer,
    row_letter: str,
    target_color: Iterable[int],
    color_wells: List[str],
    cam_index: int = 0,
    max_iterations: int = 11,
    log_cb: Callable[[str], None] | None = None,
) -> List[List[int]]:
    """Actively learn a single plate row.

    Parameters
    ----------
    robot, processor, optimizer :
        Pre-initialised helpers controlling the OT-2 and camera.
    row_letter : str
        Plate row identifier (``"A"``-``"H"``).
    target_color : iterable of int
        The RGB target colour present in column 1.
    color_wells : list of str
        Identifiers of the dye reservoirs.
    max_iterations : int, optional
        Maximum number of guess iterations.  Defaults to 11.
    log_cb : callable, optional
        If provided, called with progress log lines.

    Returns
    -------
    list of list of int
        History of volume combinations trialled.
    """

    history: List[List[int]] = []
    used_combos: set[tuple[int, ...]] = set()
    row_idx = ord(row_letter) - ord("A")

    current_iteration = 0
    while current_iteration < max_iterations:
        column = current_iteration + 2
        well_coordinate = f"{row_letter}{column}"
        if log_cb:
            log_cb(f"{row_letter} | Iter {current_iteration + 1} | Well {well_coordinate}")

        # unique recipe generation
        while True:
            volumes = optimizer.suggest_next_experiment(list(target_color))
            if tuple(volumes) not in used_combos:
                used_combos.add(tuple(volumes))
                break

        if log_cb:
            log_cb(f"Suggested: {volumes}")

        # pipette
        while True:
            try:
                for i, volume in enumerate(volumes):
                    if volume > 0:
                        robot.add_add_color_action(
                            color_well=color_wells[i],
                            plate_well=well_coordinate,
                            volume=volume,
                        )
                robot.add_mix_action(
                    plate_well=well_coordinate,
                    volume=optimizer.max_well_volume/2,
                    repetitions=3,
                )
                robot.execute_actions_on_remote()
                break
            except RuntimeError:
                if robot.last_error_type == TiprackEmptyError:
                    if log_cb:
                        log_cb("Tiprack empty - refreshing")
                    robot.add_refresh_tiprack_action()
                    robot.execute_actions_on_remote()
                else:
                    raise

        # measure
        color_data = processor.process_image(cam_index=cam_index)
        measured_color = color_data[row_idx][column - 1]
        if log_cb:
            log_cb(f"Measured: {measured_color}")

        optimizer.add_data(volumes, measured_color)
        distance = optimizer.calculate_distance(measured_color, target_color)
        if log_cb:
            log_cb(f"Distance: {distance:.2f}")

        history.append(volumes)
        if optimizer.within_tolerance(measured_color, target_color):
            if log_cb:
                log_cb(f"Matched with {volumes}")
            break

        current_iteration += 1

    return history


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_active_learning() -> None:
    """Replicates the behaviour of the original script."""

    robot = OT2Manager(
        hostname="172.26.192.201",
        username="root",
        key_filename="secret/ot2_ssh_key_remote",
        password=None,
    )

    CAM_INDEX = 2

    log_f = Path("output.txt").open("w", encoding="utf-8")
    log_f.write(
        "Row | Iter | TargetRGB | Volumes | MeasuredRGB | Dist | #Train | R2 | MSE\n"
    )
    log_f.flush()

    robot.add_blink_lights_action(num_blinks=3)
    robot.add_turn_on_lights_action()
    robot.execute_actions_on_remote()

    color_wells = ["A1", "A2", "A3"]
    plate_rows = ["A", "B", "C", "D"]

    optimizer = ColorLearningOptimizer(
        dye_count=len(color_wells),
        max_well_volume=200,
        step=1,
        tolerance=15,
        min_required_volume=20,
        exploration_weight=1.0,
        single_row_learning=True,
    )

    processor = PlateProcessor()
    color_data = processor.process_image(cam_index=CAM_INDEX)

    try:
        for row_letter in plate_rows:
            optimizer.reset()
            target_color = color_data[ord(row_letter) - ord("A")][0]
            def logger(msg: str) -> None:
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

            history = active_learn_row(
                robot,
                processor,
                optimizer,
                row_letter,
                target_color,
                color_wells,
                max_iterations=11,
                log_cb=logger,
            )
            logger(f"Completed {row_letter} in {len(history)} steps")

    finally:
        robot.add_turn_off_lights_action()
        robot.add_close_action()
        robot.execute_actions_on_remote()
        log_f.close()


if __name__ == "__main__":
    run_active_learning()

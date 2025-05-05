#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
active_color_learning.py
Row-by-row colour matching with an active-learning optimiser.
Each suggested dye recipe within a row is unique.
Logs Target RGB, Measured RGB, volumes, distance, training-set size,
plus R2 and MSE to output.txt (flushed every iteration).
"""

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from ot2_utils import TiprackEmptyError
from ot2_utils import OT2Manager
from color_learning import ColorLearningOptimizer
from camera_w_calibration import PlateProcessor

# ---------------- OT-2 connection ----------------
robot = OT2Manager(
    hostname="172.26.192.201",
    username="root",
    key_filename="secret/ot2_ssh_key_remote",
    password=None,
    virtual_mode=False,
)

log_f = None

try:
    # overwrite any previous log
    log_f = Path("output.txt").open("w", encoding="utf-8")
    log_f.write(
        "Row | Iter | TargetRGB | Volumes | MeasuredRGB | Dist | #Train | R2 | MSE\n"
    )
    log_f.flush()

    # deck lights for imaging
    robot.add_blink_lights_action(num_blinks=3)
    robot.add_turn_on_lights_action()
    robot.execute_actions_on_remote()

    # configuration
    color_slots = ["7", "8", "9"]
    plate_rows_letters = ["A", "B", "C", "D", "E", "F", "G","H"]
    # 1-index, including start and end
    rows_start = 1
    rows_end = 4
    plate_rows = plate_rows_letters[(rows_start-1):rows_end]

    MAX_WELL_VOLUME = 200
    TOLERANCE = 15
    MIN_STEP = 1
    MAX_ITERATIONS = 11

    optimizer = ColorLearningOptimizer(
        dye_count=len(color_slots),
        max_well_volume=MAX_WELL_VOLUME,
        step=MIN_STEP,
        tolerance=TOLERANCE,
        min_required_volume=20,
        optimization_mode="mlp_active",
        n_models = 5,
        exploration_weight = 1.0,
        initial_explore_count = 0,
        initial_force_all_dyes = False,
        candidate_num = 300
        single_row_learning = True,
    )

    # initial camera capture
    processor = PlateProcessor()
    color_data = processor.process_image(cam_index=0)

    # main loop
    for row_letter in plate_rows:
        print(f"\nProcessing row {row_letter}")
        optimizer.reset()
        used_combos = set()

        row_idx = ord(row_letter) - ord("A")
        target_color = color_data[row_idx][0]
        print(f"Target color: {target_color}")

        current_iteration = 0
        while current_iteration < MAX_ITERATIONS:
            column = current_iteration + 2
            well_coordinate = f"{row_letter}{column}"
            print(f"Row {row_letter} | Iter {current_iteration + 1} | Well {well_coordinate}")

            # unique recipe
            while True:
                volumes = optimizer.suggest_next_experiment(target_color)
                if tuple(volumes) not in used_combos:
                    used_combos.add(tuple(volumes))
                    break

            print(f"Suggested dye volumes: {volumes}")

            # # queue pipetting
            # for i, volume in enumerate(volumes):
            #     if volume > 0:
            #         robot.add_add_color_action(
            #             color_slot=color_slots[i],
            #             plate_well=well_coordinate,
            #             volume=volume,
            #         )
            # robot.execute_actions_on_remote()
            while True:
                try:
                    # Queue pipetting actions
                    for i, volume in enumerate(volumes):
                        if volume > 0:
                            robot.add_add_color_action(
                                color_slot=color_slots[i],
                                plate_well=well_coordinate,
                                volume=volume,
                            )
                    robot.execute_actions_on_remote()
                    break  # success, break the while loop
                except RuntimeError:
                    if robot.last_error_type == TiprackEmptyError:
                        print("Tiprack empty detected. Refreshing tiprack and retrying...")
                        robot.add_refresh_tiprack_action()
                        robot.execute_actions_on_remote()
                        print("Tiprack refreshed. Retrying pipetting...")
                    else:
                        # Some other error, not tiprack-related -> re-raise
                        raise


            # optional diffusion delay
            # time.sleep(3)

            # measure colour
            color_data = processor.process_image(cam_index=0)
            measured_color = color_data[row_idx][column - 1]
            print(f"Measured color: {measured_color}")

            # feed optimiser
            optimizer.add_data(volumes, measured_color)

            # distance to target
            distance = optimizer.calculate_distance(measured_color, target_color)
            print(f"Distance to target: {distance:.2f}")

            # metrics
            if (
                optimizer.optimization_mode == "mlp_active"
                and len(optimizer.X_train) >= 2           # same threshold used in add_data
                and all(hasattr(m, "coefs_") for m in optimizer.models)
            ):
                preds = np.mean(
                    [m.predict(optimizer.X_train) for m in optimizer.models],
                    axis=0,
                )
                mse_val = mean_squared_error(optimizer.Y_train, preds)
                r2_val = r2_score(optimizer.Y_train, preds)
            else:
                mse_val = float("nan")
                r2_val = float("nan")

            # log entry
            log_f.write(
                f"{row_letter} | {current_iteration + 1} | {target_color} | "
                f"{volumes} | {measured_color} | {distance:.2f} | "
                f"{len(optimizer.X_train)} | {r2_val:.4f} | {mse_val:.2f}\n"
            )
            log_f.flush()

            if optimizer.within_tolerance(measured_color, target_color):
                print(f"Target matched for row {row_letter}. Recipe: {volumes}")
                log_f.write(
                    f"Row {row_letter} matched with recipe {volumes}\n"
                )
                log_f.flush()
                break

            current_iteration += 1

    print("\nColour matching complete.")

except KeyboardInterrupt:
    print("\nInterrupted by user. Shutting down...")

finally:
    try:
        robot.add_turn_off_lights_action()
        robot.add_close_action()
        robot.execute_actions_on_remote()
        print("Robot shut down.")
    except Exception as e:
        print(f"Error during OT-2 shutdown: {e}")

    if log_f is not None:
        log_f.close()

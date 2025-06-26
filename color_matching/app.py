# app.py
from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
import streamlit as st
import numpy as np
from color_matching.robot.ot2_utils import OT2Manager, WellFullError, TiprackEmptyError
from camera.camera_w_calibration import PlateProcessor
import matplotlib.pyplot as plt
import time
from color_matching.active_learning.color_learning import ColorLearningOptimizer
from typing import Iterable
import itertools
from sklearn.decomposition import PCA
from GP_visualizer import plot_gp_predictions
from color_matching.data.well_data_utils import (
    load_table,
    load_global_table,
    clear_saved_tables,
    record_measurements,
    record_recipe,
    populate_optimizer,
)

def rerun() -> None:
    """Trigger a Streamlit rerun regardless of version."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()

# ——— CONFIG ———
COLOR_THRESHOLD = 20
ROWS = [chr(i) for i in range(ord("A"), ord("H")+1)]
MAX_WELLS_PER_ROW = 12
MYSTERY_COL = 1
FIRST_GUESS_COL = 2
MAX_GUESSES = MAX_WELLS_PER_ROW - 1  # 11
MIN_VOL = 20
MAX_VOL_SUM = 200
VIRTUAL_MODE = False  # set to True for virtual mode

OT_NUMBER = 4

WHITE_THRESHOLD = 120  # RGB threshold for white detection


# Example available color wells
color_wells  = ["A1", "A2", "A3"]
dye_colors = ['r', 'y', 'b']  # red, yellow, blue, for visuals only, never to be fed to the AI

FORCE_REMOTE = True  # set to True to force remote connection

# ——— info.json ———
try:
    with open(f"secret/OT_{OT_NUMBER}/info.json", "r") as f:
        info = st.session_state.get("info", {})
        info.update(json.load(f))
        st.session_state.info = info
        local_ip = info.get("local_ip", "169.254.122.0")
        local_password = info.get("local_password", "lemos")
        local_password = None if local_password == "None" else local_password

        remote_ip = info.get("remote_ip", "172.26.192.201")
        remote_password = info.get("remote_password", "None")
        remote_password = None if remote_password == "None" else remote_password

        CAM_INDEX = info.get("cam_index", 1)
except FileNotFoundError:
    st.error("Configuration file not found. Please ensure `info.json` exists in the `secret/OT_1/` directory.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding JSON in `info.json`. Please check the file format.")
    st.stop()


# ——— SESSION INIT ———
if "robot" not in st.session_state:
    if not FORCE_REMOTE:
        st.session_state.robot = OT2Manager(
            hostname=local_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key",
            password=local_password,
            reduced_tips_info=4,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE,
        )
    else:
        # fallback remote
        st.session_state.robot = OT2Manager(
            hostname=remote_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key_remote",
            password=remote_password,
            reduced_tips_info=4,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE
        )

    st.session_state.robot.add_turn_on_lights_action()
    st.session_state.robot.execute_actions_on_remote()

    print("Lights turned on.")
    time.sleep(2)  # wait for lights to stabilize

st.session_state.processor = PlateProcessor(virtual_mode=VIRTUAL_MODE)
st.session_state.setdefault("well_data", load_table())
st.session_state.setdefault("global_well_data", load_global_table())

# track how many guesses we've made per row, and the measured RGBs+distances
for row in ROWS:
    st.session_state.setdefault(f"guesses_{row}", 0)
    st.session_state.setdefault(f"history_{row}", [])  # list of (rgb, distance)

# State for AI runs
st.session_state.setdefault("ai_running", False)
st.session_state.setdefault("ai_row", None)
st.session_state.setdefault("ai_iter", 0)
st.session_state.setdefault("ai_optimizer", None)
st.session_state.setdefault("ai_used_combos", set())
st.session_state.setdefault("ai_target", None)
st.session_state.setdefault("ai_log", [])
st.session_state.setdefault("ai_step_pending", False)

# Initialize AI optimizer if not already set
#x_history = st.session_state.ai_optimizer.X_train.copy()
#y_history = st.session_state.ai_optimizer.Y_train.copy()
#print(f"AI history: {len(x_history)} samples, {len(y_history)} colors")
#
#st.session_state.ai_optimizer = None
if st.session_state.ai_optimizer is None:
    st.session_state.ai_optimizer = ColorLearningOptimizer(
            dye_count=len(color_wells ),
            tolerance=COLOR_THRESHOLD,
            single_row_learning=False,
        )
    populate_optimizer(
        st.session_state.global_well_data,
        st.session_state.ai_optimizer,
        restore=True,
    )
    #st.session_state.ai_optimizer.X_train = x_history
    #st.session_state.ai_optimizer.Y_train = y_history

# ——— UI ———
st.title("🎨 Human vs Robot: Color-Mixing Challenge")
st.markdown(
    """
1. Mystery color is in column 1 of the selected row.  
2. Enter a “recipe” (volumes for each color slot) and click **Make recipe**.  
3. The robot will pipette it, we'll photograph it, and you'll see your guess's RGB + distance from target.  
"""
)

row = st.selectbox("Select row", ROWS, disabled=st.session_state.ai_running)

# ——— ROW SWITCH: re-read plate & prepopulate history ———
if st.session_state.get("last_row") != row:
    st.session_state.last_row = row

    # reset counters & history
    st.session_state[f"guesses_{row}"] = 0
    st.session_state[f"history_{row}"] = []

    # snap the whole plate
    full_plate = st.session_state.processor.process_image(
        cam_index=CAM_INDEX,
        calib=f"secret/OT_{OT_NUMBER}/calibration.json",
    )
    record_measurements(
        st.session_state.well_data,
        st.session_state.global_well_data,
        full_plate,
    )

    # extract the mystery target
    r_m, g_m, b_m = full_plate[ord(row) - ord("A")][MYSTERY_COL - 1]
    myst_rgb = np.array([r_m, g_m, b_m])
    st.session_state[f"mystery_rgb_{row}"] = myst_rgb.tolist()

    # now scan existing guesses in cols 2→12
    prehist = []
    for col in range(FIRST_GUESS_COL - 1, MAX_WELLS_PER_ROW):
        rgb = full_plate[ord(row) - ord("A")][col]
        # white threshold:
        if any(ch < WHITE_THRESHOLD for ch in rgb):
            dist = float(np.linalg.norm(rgb.astype(float) - myst_rgb))
            prehist.append((rgb.tolist(), dist))

    # seed session_state
    st.session_state[f"history_{row}"] = prehist
    st.session_state[f"guesses_{row}"]  = len(prehist)

st.markdown(f"**Mystery color (Col {MYSTERY_COL}) for row {row}:**")
r, g, b = st.session_state[f"mystery_rgb_{row}"]
st.markdown(
    f'<div style="display:inline-block;width:30px;height:30px;'
    f'background-color:rgb({r},{g},{b});border:1px solid #000;"></div>',
    unsafe_allow_html=True,
)

st.subheader("Build your recipe")
cols = st.columns(len(color_wells ))
volumes = {}
for i, slot in enumerate(color_wells ):
    with cols[i]:
        volumes[slot] = st.number_input(
            f"Slot {slot}",
            min_value=0.0,
            step=1.0,
            key=f"vol_{row}_{slot}",
            disabled=st.session_state.ai_running
        )

# Validation
total = sum(volumes.values())
ok_sizes = all(v == 0 or v >= MIN_VOL for v in volumes.values())
ok_sum = total <= MAX_VOL_SUM and total > 0
guesses_left = st.session_state[f"guesses_{row}"] < MAX_GUESSES

if not ok_sizes:
    st.warning(f"Each non-zero volume must be ≥ {MIN_VOL} µL.")
if not ok_sum:
    st.warning(f"Total volume must be between {MIN_VOL} µL and {MAX_VOL_SUM} µL.")
if not guesses_left:
    st.info("You've used up all 11 guesses in this row.")

can_make = ok_sizes and ok_sum and guesses_left
make_btn = st.button(
    "Make recipe",
    disabled=(not can_make) or st.session_state.ai_running,
)
close_btn = st.button("Close robot", disabled=st.session_state.ai_running)
deploy_btn = False
if st.session_state[f"guesses_{row}"] == 0:
    deploy_btn = st.button(
        "Deploy AI",
        disabled=st.session_state.ai_running,
    )

stop_ai_btn = False
if st.session_state.ai_running:
    stop_ai_btn = st.button(
        "Stop AI",
        disabled=not st.session_state.ai_running,
    )
    if stop_ai_btn:
        st.session_state.ai_running = False
        st.session_state.ai_step_pending = False
        rerun()

reset_btn = st.button("Reset model knowledge", disabled=st.session_state.ai_running)
restore_btn = st.button(
    "Restore model knowledge",
    disabled=st.session_state.ai_running,
)
clear_btn = st.button("Clear saved recipes", disabled=st.session_state.ai_running)

# ——— ACTION ———
if make_btn:
    guess_index = st.session_state[f"guesses_{row}"] + 1  # 1–11
    target_well = f"{row}{FIRST_GUESS_COL + guess_index - 1}"

    robot = st.session_state.robot
    
    total_vol = 0
    for slot, vol in volumes.items():
        if vol > 0:
            robot.add_add_color_action(tip_ID=slot,
                                       plate_well=target_well,
                                       volume=int(vol))
            total_vol += vol
    robot.add_mix_action(
        plate_well=target_well,
        volume=total_vol/2,
        repetitions=3
    )
    try:
        robot.execute_actions_on_remote()
    except Exception as e:
        st.error(f"Robot error: {e}")
        st.stop()

    # photo & measure
    full_plate = st.session_state.processor.process_image(
        cam_index=CAM_INDEX,
        calib=f"secret/OT_{OT_NUMBER}/calibration.json",
    )
    record_measurements(
        st.session_state.well_data,
        st.session_state.global_well_data,
        full_plate,
    )
    # for debug: plot the full plate and all colors
    #st.image(full_plate, caption="Full plate image", use_column_width=True)
    row_idx = ord(row) - ord("A")
    rgb = full_plate[row_idx, FIRST_GUESS_COL + guess_index - 2]
    myst = np.array(st.session_state[f"mystery_rgb_{row}"])
    dist = float(np.linalg.norm(np.array(rgb) - myst))

    # store history
    hist = st.session_state[f"history_{row}"]
    hist.append((rgb.tolist(), dist))
    st.session_state[f"guesses_{row}"] += 1
    record_recipe(
        st.session_state.well_data,
        st.session_state.global_well_data,
        target_well,
        [int(volumes[s]) for s in color_wells],
    )
    rerun()

if deploy_btn:
    st.session_state.well_data = load_table()
    st.session_state.global_well_data = load_global_table()
    st.session_state.ai_target = st.session_state.well_data[
        f"{row}{MYSTERY_COL}"
    ]["rgb"]
    st.session_state.ai_running = True
    st.session_state.ai_row = row
    st.session_state.ai_iter = 0
    st.session_state.ai_used_combos = set()
    st.session_state.ai_log = []
    st.session_state.ai_step_pending = True
    rerun()

if close_btn:
    try:
        st.session_state.robot.add_turn_off_lights_action()
        st.session_state.robot.add_close_action()
        st.session_state.robot.execute_actions_on_remote()
    except Exception as e:
        st.error(f"Robot error: {e}")
        st.stop()

if reset_btn:
    st.session_state.ai_optimizer.reset()
    st.rerun()
if restore_btn:
    populate_optimizer(
        st.session_state.global_well_data,
        st.session_state.ai_optimizer,
        restore=True,
    )
    st.rerun()
if clear_btn:
    st.session_state.well_data = clear_saved_tables()
    st.session_state.global_well_data = st.session_state.well_data.copy()
    st.session_state.ai_optimizer.reset()
    st.session_state.ai_optimizer.X_train_permanent = []
    st.session_state.ai_optimizer.Y_train_permanent = []
    st.rerun()

# ——— DISPLAY HISTORY ———
if st.session_state[f"history_{row}"]:
    st.subheader("Your guesses so far")
    row_hist = st.session_state[f"history_{row}"]
    # show colored squares
    html = ""
    for (rgb, dist) in row_hist:
        r, g, b = rgb
        box = (
            f'<div style="display:inline-block;width:20px;height:20px;'
            f'background-color:rgb({r},{g},{b});border:1px solid #333;'
            f'margin-right:4px"></div>'
        )
        html += box
    st.markdown(html, unsafe_allow_html=True)

    # last distance & threshold
    last_dist = row_hist[-1][1]
    color = "#d4d4d4" if last_dist < COLOR_THRESHOLD else "#f87474"
    st.markdown(
        f'<div style="padding:8px;background-color:{color};'
        f'border-radius:4px">Distance from target: {last_dist:.1f}</div>',
        unsafe_allow_html=True,
    )

    # add a plot as distance vs. guess number
    # pull out the distances
    dists = [dist for (_, dist) in row_hist]

    # build a simple line plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(dists) + 1), dists, marker='o')
    ax.axhline(COLOR_THRESHOLD, linestyle='--')  # threshold line
    ax.set_xlabel("Guess #")
    ax.set_ylabel("Distance")
    ax.set_title(f"Distance vs. Guess for row {row}")

    st.pyplot(fig)

# ——— MODEL PLOT ———
grid = np.linspace(0, MAX_VOL_SUM, 200)

# Prepare a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

Y_train = np.array(st.session_state.ai_optimizer.Y_train)
X_train = np.array(st.session_state.ai_optimizer.X_train)

if len(Y_train) > 0:
    fig = plot_gp_predictions(
        gp_models=st.session_state.ai_optimizer.models,
        X_train=X_train,
        Y_train=Y_train
    )
    st.pyplot(fig)


if st.session_state.ai_log:
    st.subheader("AI Log")
    st.text("\n".join(str(x) for x in st.session_state.ai_log))


# ——— AI STEP ———  

def _ai_step() -> None:
    """Execute one iteration of the active learning loop."""
    row_letter: str = st.session_state.ai_row
    optimizer: ColorLearningOptimizer = st.session_state.ai_optimizer
    robot = st.session_state.robot
    processor = st.session_state.processor
    st.session_state.well_data = load_table()
    st.session_state.global_well_data = load_global_table()
    target_color: Iterable[int] = st.session_state.well_data[
        f"{row_letter}{MYSTERY_COL}"
    ]["rgb"]
    st.session_state.ai_target = target_color
    iteration: int = st.session_state.ai_iter
    used: set = st.session_state.ai_used_combos

    column = iteration + 2
    well_coordinate = f"{row_letter}{column}"
    row_idx = ord(row_letter) - ord("A")

    st.session_state.ai_log.append(
        f"{row_letter} | Iter {iteration + 1} | Well {well_coordinate}"
    )

    # Update the exploration weight based on the number of guesses
    if iteration < len(color_wells ):
        #optimizer.update_exploration_weight(0.4)
        exploration_weight = 1
    else:
        # Exploration weight should decrease as we get closer to the final well
        # When idx is 11, the exploration weight should be 0
        exploration_weight = max(0.0, 1 - ((iteration) / (MAX_GUESSES)))
        #exploration_weight = 0.0
        optimizer.update_exploration_weight(exploration_weight)

    attempts_remaining = 3
    while attempts_remaining > 0:
        attempts_remaining -= 1
        vols = optimizer.suggest_next_experiment(list(target_color))
        if tuple(vols) not in used:
            used.add(tuple(vols))
            break

    st.session_state.ai_used_combos = used
    st.session_state.ai_log.append(f"Suggested: {vols}")

    while True:
        try:
            for i, volume in enumerate(vols):
                if volume > 0:
                    robot.add_add_color_action(
                        color_well=color_wells[i],
                        plate_well=well_coordinate,
                        volume=volume,
                    )
            robot.add_mix_action(
                plate_well=well_coordinate,
                volume=optimizer.max_well_volume / 2,
                repetitions=3,
            )
            robot.execute_actions_on_remote()
            break
        except RuntimeError:
            if robot.last_error_type == TiprackEmptyError:
                st.session_state.ai_log.append("Tiprack empty - refreshing")
                robot.add_refresh_tiprack_action()
                robot.execute_actions_on_remote()
            else:
                raise

    color_data = processor.process_image(
        cam_index=CAM_INDEX,
        calib=f"secret/OT_{OT_NUMBER}/calibration.json",
    )
    record_measurements(
        st.session_state.well_data,
        st.session_state.global_well_data,
        color_data,
    )
    measured_color = color_data[row_idx][column - 1]
    st.session_state.ai_log.append(f"Measured: {measured_color}")

    optimizer.add_data(vols, measured_color)
    distance = optimizer.calculate_distance(measured_color, target_color)
    st.session_state.ai_log.append(f"Distance: {distance:.2f}")

    hist = st.session_state[f"history_{row_letter}"]
    hist.append((measured_color.tolist(), float(distance)))
    st.session_state[f"guesses_{row_letter}"] += 1
    record_recipe(
        st.session_state.well_data,
        st.session_state.global_well_data,
        well_coordinate,
        vols,
    )

    if optimizer.within_tolerance(measured_color, target_color):
        st.session_state.ai_log.append(f"Matched with {vols}")
        st.session_state.ai_running = False
    else:
        st.session_state.ai_iter += 1
        if st.session_state.ai_iter >= MAX_GUESSES:
            print("AI exhausted all guesses without matching.")
            st.session_state.ai_running = False

print(st.session_state.ai_optimizer.X_train)
print(st.session_state.ai_optimizer.Y_train)

if st.session_state.ai_running or st.session_state.ai_step_pending:
    st.session_state.ai_step_pending = False
    _ai_step()
    if st.session_state.ai_running:
        st.session_state.ai_step_pending = True
    else:
        st.success("AI has found a matching recipe or exhausted guesses.")
    rerun()

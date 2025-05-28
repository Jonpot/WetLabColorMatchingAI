# app.py
import json
import streamlit as st
import numpy as np
from robot.ot2_utils import OT2Manager, WellFullError, TiprackEmptyError
from camera.camera_w_calibration import PlateProcessor
import matplotlib.pyplot as plt
import time
from active_learning.color_learning import ColorLearningOptimizer
from typing import Iterable
import itertools


def rerun() -> None:
    """Trigger a Streamlit rerun regardless of version."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()

# ——— CONFIG ———
COLOR_THRESHOLD = 60
ROWS = [chr(i) for i in range(ord("A"), ord("H")+1)]
MAX_WELLS_PER_ROW = 12
MYSTERY_COL = 1
FIRST_GUESS_COL = 2
MAX_GUESSES = MAX_WELLS_PER_ROW - 1  # 11
MIN_VOL = 20
MAX_VOL_SUM = 200
VIRTUAL_MODE = False  # set to True for virtual mode

OT_NUMBER = 4



# Example available color slots
color_slots = ["7", "8", "9"]

FORCE_REMOTE = False  # set to True to force remote connection

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


def _ai_step() -> None:
    """Execute one iteration of the active learning loop."""
    row_letter: str = st.session_state.ai_row
    optimizer: ColorLearningOptimizer = st.session_state.ai_optimizer
    robot = st.session_state.robot
    processor = st.session_state.processor
    target_color: Iterable[int] = st.session_state.ai_target
    iteration: int = st.session_state.ai_iter
    used: set = st.session_state.ai_used_combos

    column = iteration + 2
    well_coordinate = f"{row_letter}{column}"
    row_idx = ord(row_letter) - ord("A")

    st.session_state.ai_log.append(
        f"{row_letter} | Iter {iteration + 1} | Well {well_coordinate}"
    )

    while True:
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
                        color_slot=color_slots[i],
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

    color_data = processor.process_image(cam_index=CAM_INDEX,
                                         calib=f"secret/OT_{OT_NUMBER}/calibration.json")
    measured_color = color_data[row_idx][column - 1]
    st.session_state.ai_log.append(f"Measured: {measured_color}")

    optimizer.add_data(vols, measured_color)
    distance = optimizer.calculate_distance(measured_color, target_color)
    st.session_state.ai_log.append(f"Distance: {distance:.2f}")

    hist = st.session_state[f"history_{row_letter}"]
    hist.append((measured_color.tolist(), float(distance)))
    st.session_state[f"guesses_{row_letter}"] += 1

    if optimizer.within_tolerance(measured_color, target_color):
        st.session_state.ai_log.append(f"Matched with {vols}")
        st.session_state.ai_running = False
    else:
        st.session_state.ai_iter += 1
        if st.session_state.ai_iter >= MAX_GUESSES:
            st.session_state.ai_running = False


if st.session_state.ai_running and st.session_state.ai_step_pending:
    st.session_state.ai_step_pending = False
    _ai_step()
    if st.session_state.ai_running:
        st.session_state.ai_step_pending = True
        rerun()


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
    full_plate = st.session_state.processor.process_image(cam_index=CAM_INDEX,
                                                          calib=f"secret/OT_{OT_NUMBER}/calibration.json")

    # extract the mystery target
    r_m, g_m, b_m = full_plate[ord(row) - ord("A")][MYSTERY_COL - 1]
    myst_rgb = np.array([r_m, g_m, b_m])
    st.session_state[f"mystery_rgb_{row}"] = myst_rgb.tolist()

    # now scan existing guesses in cols 2→12
    prehist = []
    for col in range(FIRST_GUESS_COL - 1, MAX_WELLS_PER_ROW):
        rgb = full_plate[ord(row) - ord("A")][col]
        # white threshold:
        if any(ch < 180 for ch in rgb):
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
cols = st.columns(len(color_slots))
volumes = {}
for i, slot in enumerate(color_slots):
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

# ——— ACTION ———
if make_btn:
    guess_index = st.session_state[f"guesses_{row}"] + 1  # 1–11
    target_well = f"{row}{FIRST_GUESS_COL + guess_index - 1}"

    robot = st.session_state.robot
    
    total_vol = 0
    for slot, vol in volumes.items():
        if vol > 0:
            robot.add_add_color_action(color_slot=slot,
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
    full_plate = st.session_state.processor.process_image(cam_index=CAM_INDEX,
                                                          calib=f"secret/OT_{OT_NUMBER}/calibration.json")
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
    rerun()

if deploy_btn:
    st.session_state.ai_running = True
    st.session_state.ai_row = row
    st.session_state.ai_iter = 0
    st.session_state.ai_optimizer = ColorLearningOptimizer(
        dye_count=len(color_slots),
        tolerance=COLOR_THRESHOLD,
    )
    st.session_state.ai_used_combos = set()
    st.session_state.ai_target = st.session_state[f"mystery_rgb_{row}"]
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
    color = "#d4f8d4" if last_dist < COLOR_THRESHOLD else "#f87474"
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

if st.session_state.ai_log:
    st.subheader("AI Log")
    st.text("\n".join(str(x) for x in st.session_state.ai_log))
    
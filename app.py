# app.py
import streamlit as st
import numpy as np
from ot2_utils import OT2Manager, WellFullError, TiprackEmptyError
from camera_w_calibration import PlateProcessor

# ——— CONFIG ———
COLOR_THRESHOLD = 30
ROWS = [chr(i) for i in range(ord("A"), ord("H")+1)]
MAX_WELLS_PER_ROW = 12
MYSTERY_COL = 1
FIRST_GUESS_COL = 2
MAX_GUESSES = MAX_WELLS_PER_ROW - 1  # 11
MIN_VOL = 20
MAX_VOL_SUM = 200

# Example available color slots
color_slots = ["7", "8", "9"]


# ——— SESSION INIT ———
if "robot" not in st.session_state:
    try:
        st.session_state.robot = OT2Manager(
            hostname="169.254.122.0",
            username="root",
            key_filename="secret/ot2_ssh_key",
            password="lemos",
            reduced_tips_info=3,
            virtual_mode=True
        )
    except Exception:
        # fallback remote
        st.session_state.robot = OT2Manager(
            hostname="172.26.192.201",
            username="root",
            key_filename="secret/ot2_ssh_key_remote",
            password=None,
        )
    st.session_state.processor = PlateProcessor()

# track how many guesses we've made per row, and the measured RGBs+distances
for row in ROWS:
    st.session_state.setdefault(f"guesses_{row}", 0)
    st.session_state.setdefault(f"history_{row}", [])  # list of (rgb, distance)

# ——— UI ———
st.title("🎨 Human vs Robot: Color-Mixing Challenge")
st.markdown(
    """
1. Mystery color is in column 1 of the selected row.  
2. Enter a “recipe” (volumes for each color slot) and click **Make recipe**.  
3. The robot will pipette it, we’ll photograph it, and you’ll see your guess’s RGB + distance from target.  
"""
)

row = st.selectbox("Select row", ROWS)

# Reset row display (re-read camera) if they switch rows
if st.session_state.get("last_row") != row:
    st.session_state.last_row = row
    # read full plate once so we know the mystery color
    full_plate = st.session_state.processor.process_image(cam_index=0)
    # extract the target/mystery color for this row
    myst_rgb = full_plate[ord(row) - ord("A"), MYSTERY_COL - 1]
    st.session_state[f"mystery_rgb_{row}"] = myst_rgb.tolist()

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
            key=f"vol_{row}_{slot}"
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
    st.info("You’ve used up all 11 guesses in this row.")

can_make = ok_sizes and ok_sum and guesses_left
make_btn = st.button("Make recipe", disabled=not can_make)

# ——— ACTION ———
if make_btn:
    guess_index = st.session_state[f"guesses_{row}"] + 1  # 1–11
    target_well = f"{row}{FIRST_GUESS_COL + guess_index - 1}"

    robot = st.session_state.robot
    robot.add_turn_on_lights_action()
    for slot, vol in volumes.items():
        if vol > 0:
            robot.add_add_color_action(color_slot=slot,
                                       plate_well=target_well,
                                       volume=int(vol))
    robot.add_turn_off_lights_action()
    robot.add_close_action()

    try:
        robot.execute_actions_on_remote()
    except Exception as e:
        st.error(f"Robot error: {e}")
        st.stop()

    # photo & measure
    full_plate = st.session_state.processor.process_image(cam_index=0)
    row_idx = ord(row) - ord("A")
    rgb = full_plate[row_idx, FIRST_GUESS_COL + guess_index - 2]
    myst = np.array(st.session_state[f"mystery_rgb_{row}"])
    dist = float(np.linalg.norm(np.array(rgb) - myst))

    # store history
    hist = st.session_state[f"history_{row}"]
    hist.append((rgb.tolist(), dist))
    st.session_state[f"guesses_{row}"] += 1

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
    color = "#d4f8d4" if last_dist < COLOR_THRESHOLD else "#f8d4d4"
    st.markdown(
        f'<div style="padding:8px;background-color:{color};'
        f'border-radius:4px">Distance from target: **{last_dist:.1f}**</div>',
        unsafe_allow_html=True,
    )

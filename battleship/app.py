import json
from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from string import ascii_uppercase
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from battleship.ai import AI
from battleship.plate_state_processor import PlateStateProcessor, WellState
from battleship.robot.ot2_utils import OT2Manager


CONFIG_PATH = Path(__file__).resolve().parent / "configuration.json"


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"plate_schema": {"rows": 8, "columns": 12}, "ship_schema": {}}


def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


OT_NUMBER = 1
VIRTUAL_MODE = True
FORCE_REMOTE = False

# ---- info.json ----
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
        CAM_INDEX = info.get("cam_index", 2)
except FileNotFoundError:
    st.error("Configuration file not found. Please ensure `info.json` exists in the `secret/OT_1/` directory.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding JSON in `info.json`. Please check the file format.")
    st.stop()




def plot_board(board: np.ndarray) -> plt.Figure:
    cmap = {
        WellState.UNKNOWN: "#d9d9d9",
        WellState.MISS: "#6fa8dc",
        WellState.HIT: "#e06666",
    }
    rows, cols = board.shape
    mapping = {state.value: idx for idx, state in enumerate(cmap)}
    data = np.vectorize(lambda x: mapping[x.value])(board).astype(int)
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=plt.matplotlib.colors.ListedColormap(list(cmap.values())), vmin=0, vmax=2)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(i + 1) for i in range(cols)])
    ax.set_yticks(range(rows))
    ax.set_yticklabels(list(ascii_uppercase[:rows]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Board state")
    ax.grid(True, which="both", color="black", linewidth=1, linestyle="--")
    return fig


config = load_config()

if "robot" not in st.session_state:
    if not FORCE_REMOTE:
        st.session_state.robot = OT2Manager(
            hostname=local_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key",
            password=local_password,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE,
        )
    else:
        st.session_state.robot = OT2Manager(
            hostname=remote_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key_remote",
            password=remote_password,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE,
        )
    st.session_state.robot.add_turn_on_lights_action()
    st.session_state.robot.execute_actions_on_remote()
    st.session_state.processor = PlateStateProcessor(config.get("plate_schema", {}), cam_index=CAM_INDEX, virtual_mode=VIRTUAL_MODE)

st.title("Battleship Robot")

plate_cfg = config.get("plate_schema", {})
ship_cfg = config.get("ship_schema", {})

with st.expander("Configuration", expanded=True):
    st.subheader("Plate")
    rows = st.number_input("Rows", min_value=1, value=int(plate_cfg.get("rows", 8)), step=1)
    cols = st.number_input("Columns", min_value=1, value=int(plate_cfg.get("columns", 12)), step=1)

    st.subheader("Ships")
    for name, val in list(ship_cfg.items()):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{name}**")
        with col2:
            ship_cfg[name]["length"] = st.number_input(
                f"Length for {name}", min_value=1, value=int(val["length"]), key=f"len_{name}")
        with col3:
            ship_cfg[name]["count"] = st.number_input(
                f"Count for {name}", min_value=1, value=int(val["count"]), key=f"cnt_{name}")
    with st.form("add_ship"):
        st.markdown("### Add ship")
        new_name = st.text_input("Name")
        new_len = st.number_input("Length", min_value=1, value=1)
        new_cnt = st.number_input("Count", min_value=1, value=1)
        if st.form_submit_button("Add") and new_name:
            ship_cfg[new_name] = {"length": int(new_len), "count": int(new_cnt)}
    if st.button("Save configuration"):
        config["plate_schema"] = {"rows": int(rows), "columns": int(cols)}
        config["ship_schema"] = ship_cfg
        save_config(config)
        st.success("Configuration saved")


def run_ai(cfg: Dict[str, Any], processor: PlateStateProcessor, robot: OT2Manager) -> None:
    ships = cfg["ship_schema"]
    ai = AI(processor, robot, ships)

    placeholder = st.empty()
    step = 0
    while True:
        if ai.determine_game_state():
            break
        try:
            move = ai.get_next_move()
        except RuntimeError as e:
            print(f"Error determining next move: {e}")
            break
        ai.fire_missile(move)
        ai.board_history.append(ai.board_state.copy())
        step += 1
        with placeholder.container():
            st.subheader(f"Step {step}: fired at {ascii_uppercase[move[0]]}{move[1] + 1}")
            fig = plot_board(ai.board_state)
            st.pyplot(fig)
    st.success("Game over")
    fig = plot_board(ai.board_state)
    st.pyplot(fig)


if st.button("Start AI"):
    run_ai(config, st.session_state.processor, st.session_state.robot)

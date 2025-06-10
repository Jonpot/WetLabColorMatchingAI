import json
from pathlib import Path
import sys
import importlib
import inspect
import pkgutil
import time

# --- Add project root to path ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from string import ascii_uppercase
from typing import Any, Dict, List, Type

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# --- Import Battleship Framework Components ---
from battleship.game_manager import BattleshipGame
from battleship.plate_state_processor import DualPlateStateProcessor, WellState
from battleship.robot.ot2_utils import OT2Manager
from battleship.ai.base_ai import BattleshipAI
from battleship.ai.probabilistic_ai import ProbabilisticAI # Default AI

# --- App Configuration ---
CONFIG_PATH = Path(__file__).resolve().parent / "configuration.json"
OT_NUMBER = 2
VIRTUAL_MODE = False # Set to True to run without a robot
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



def load_config() -> Dict[str, Any]:
    """Loads plate and ship configuration from a JSON file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    # Default config if none exists
    return {
        "plate_schema": {"rows": 8, "columns": 12},
        "ship_schema": {
            "Carrier": {"length": 5, "count": 1},
            "Battleship": {"length": 4, "count": 1},
            "Cruiser": {"length": 3, "count": 1},
        }
    }


def save_config(cfg: Dict[str, Any]) -> None:
    """Saves configuration to a JSON file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


def find_ai_classes() -> Dict[str, Type[BattleshipAI]]:
    """Dynamically finds all subclasses of BattleshipAI in the 'battleship.ai' module."""
    ai_classes = {"ProbabilisticAI": ProbabilisticAI} # Start with the default
    ai_module_path = Path(__file__).resolve().parent.parent / "battleship" / "ai"
    
    for _, module_name, _ in pkgutil.iter_modules([str(ai_module_path)]):
        if module_name not in ["base_ai", "probabilistic_ai"]:
            try:
                module = importlib.import_module(f"battleship.ai.{module_name}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BattleshipAI) and obj is not BattleshipAI:
                        ai_classes[name] = obj
            except Exception as e:
                st.warning(f"Could not load AI from {module_name}: {e}")
    return ai_classes


def plot_board(board: np.ndarray, title: str) -> plt.Figure:
    """Creates a matplotlib figure of the game board."""
    cmap = {
        WellState.UNKNOWN: "#d9d9d9",
        WellState.MISS: "#6fa8dc",
        WellState.HIT: "#e06666",
    }
    rows, cols = board.shape
    data = np.array([[state.value for state in row] for row in board])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(data, cmap=plt.matplotlib.colors.ListedColormap(list(cmap.values())), vmin=0, vmax=2)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels([str(i + 1) for i in range(cols)])
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(list(ascii_uppercase[:rows]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.set_xticks(np.arange(cols+1)-.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    fig.tight_layout()
    return fig

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("🚢 Battleship AI Competition 🚢")


config = load_config()

# --- Initialize Robot and Processors in Session State ---
if "robot" not in st.session_state:
    if not FORCE_REMOTE:
        st.session_state.robot = OT2Manager(
            hostname=local_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key",
            password=local_password,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE,
            reduced_tips_info=2
        )
    else:
        st.session_state.robot = OT2Manager(
            hostname=remote_ip,
            username="root",
            key_filename=f"secret/OT_{OT_NUMBER}/ot2_ssh_key_remote",
            password=remote_password,
            bypass_startup_key=True,
            virtual_mode=VIRTUAL_MODE,
            reduced_tips_info=2
        )
    st.session_state.robot.add_turn_on_lights_action()
    st.session_state.robot.execute_actions_on_remote()
    st.session_state.processor = DualPlateStateProcessor(config.get("plate_schema", {}), ot_number=OT_NUMBER, cam_index=CAM_INDEX, virtual_mode=VIRTUAL_MODE)


# --- UI Layout ---
with st.sidebar:
    st.header("Configuration")
    with st.expander("Plate & Ship Setup", expanded=False):
        st.subheader("Plate")
        rows = st.number_input("Rows", min_value=1, value=int(config["plate_schema"].get("rows", 8)), step=1)
        cols = st.number_input("Columns", min_value=1, value=int(config["plate_schema"].get("columns", 12)), step=1)

        st.subheader("Ships")
        # Display and allow modification of existing ships
        # (Your ship configuration UI can be pasted here)

        if st.button("Save Configuration"):
            config["plate_schema"] = {"rows": int(rows), "columns": int(cols)}
            # config["ship_schema"] = ship_cfg # Update with ship data from UI
            save_config(config)
            st.success("Configuration saved!")
            st.rerun()

    st.header("Player Setup")
    available_ais = find_ai_classes()
    ai_names = list(available_ais.keys())
    
    p1_ai_choice = st.selectbox("Select AI for Player 1", options=ai_names, index=0)
    p2_ai_choice = st.selectbox("Select AI for Player 2", options=ai_names, index=ai_names.index("ProbabilisticAI") if "ProbabilisticAI" in ai_names else 0)

    start_button = st.button("🚀 Start Competition!", type="primary", use_container_width=True)

# --- Game Display Area ---
st.header("Game Boards")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Player 1 Board")
    board_placeholder_1 = st.empty()
with col2:
    st.subheader("Player 2 Board")
    board_placeholder_2 = st.empty()
status_placeholder = st.empty()
log_placeholder = st.empty()

if start_button:
    # --- Game Initialization ---
    plate_shape = (config["plate_schema"]["rows"], config["plate_schema"]["columns"])
    ship_schema = config["ship_schema"]

    Player1AI = available_ais[p1_ai_choice]
    Player2AI = available_ais[p2_ai_choice]

    player_1 = Player1AI("player_1", plate_shape, ship_schema)
    player_2 = Player2AI("player_2", plate_shape, ship_schema)

    game = BattleshipGame(player_1, player_2, st.session_state.processor, st.session_state.robot)

    # --- Live Game Loop ---
    # Initial board display
    board_placeholder_1.pyplot(plot_board(player_1.board_state, f"Player 1: {p1_ai_choice}"))
    board_placeholder_2.pyplot(plot_board(player_2.board_state, f"Player 2: {p2_ai_choice}"))
    status_placeholder.info("Game starting...")
    time.sleep(2) # Pause to show initial empty boards

    winner = None
    for state in game.run_game_live():
        # Update status message
        status_text = f"**Turn {state['turn']}**: {state['active_player']} fires at **{state['move']}**... It's a **{state['result']}**!"
        status_placeholder.markdown(status_text, unsafe_allow_html=True)
        
        # Update boards
        board_placeholder_1.pyplot(plot_board(state['board_p1'], f"Player 1: {p1_ai_choice}"))
        board_placeholder_2.pyplot(plot_board(state['board_p2'], f"Player 2: {p2_ai_choice}"))
        
        # Update history log
        history_df = pd.DataFrame(state['history']).set_index('turn')
        with log_placeholder.container():
            st.write("--- Game Log ---")
            st.dataframe(history_df, use_container_width=True)
        
        if state['winner']:
            winner = state['winner']
            break # Exit the loop once a winner is found
            
        time.sleep(1.0) # Pause between moves to make it watchable

    if winner:
        winner_name = p1_ai_choice if winner == 'player_1' else p2_ai_choice
        status_placeholder.success(f"## 🎉 GAME OVER! {winner} ({winner_name}) wins! 🎉")
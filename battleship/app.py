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
from typing import Any, Dict, List, Type, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# --- Import Battleship Framework Components ---
from battleship.game_manager import BattleshipGame
from battleship.plate_state_processor import DualPlateStateProcessor, WellState
from battleship.robot.ot2_utils import OT2Manager
from battleship.ai.base_ai import BattleshipAI
from battleship.ai.probabilistic_ai import ProbabilisticAI  # Default AI
from battleship.placement_ai import PlacementAI, NaivePlacementAI
from battleship import placement_ai
from battleship.placement_utils import validate_placement_schema, coords_from_schema

# --- App Configuration ---
CONFIG_PATH = Path(__file__).resolve().parent / "configuration.json"
OT_NUMBER = 4
VIRTUAL_MODE = False # Set to True to run without a robot
FORCE_REMOTE = True

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


def find_placement_ai_classes() -> Dict[str, Type[PlacementAI]]:
    classes: Dict[str, Type[PlacementAI]] = {"NaivePlacementAI": NaivePlacementAI}
    module_path = Path(__file__).resolve().parent.parent / "battleship" / "placement_ai"
    for _, module_name, _ in pkgutil.iter_modules([str(module_path)]):
        if module_name not in ["base_placement_ai", "naive_placement_ai", "__init__", "random_placement_ai"]:
            try:
                module = importlib.import_module(f"battleship.placement_ai.{module_name}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, PlacementAI) and obj is not PlacementAI:
                        classes[name] = obj
            except Exception as e:
                st.warning(f"Could not load Placement AI from {module_name}: {e}")
    # ensure RandomPlacementAI is included from package init
    classes.setdefault("RandomPlacementAI", placement_ai.RandomPlacementAI)
    return classes


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


def plot_ship_placement(board_shape: Tuple[int, int], placement: List[Dict[str, Any]]) -> plt.Figure:
    rows, cols = board_shape
    board = np.zeros(board_shape)
    for item in placement:
        r, c, l = item['row'], item['col'], item['length']
        if item['direction'] == 'horizontal':
            board[r, c:c+l] = 1
        else:
            board[r:r+l, c] = 1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(board, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels([str(i + 1) for i in range(cols)])
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(list(ascii_uppercase[:rows]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Ship Placement", fontsize=16, pad=20)
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
plate_shape_full = (
    config["plate_schema"].get("rows", 8),
    config["plate_schema"].get("columns", 12),
)
game_shape = (plate_shape_full[0], plate_shape_full[1] - 1)
ship_schema = config.get("ship_schema", {})

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
    st.session_state.placement = {1: None, 2: None}
    st.session_state.liquids_placed = {1: False, 2: False}
    st.session_state.game_ai_choice = {1: None, 2: None}


# --- UI Layout ---
with st.sidebar:
    st.header("Configuration")
    with st.expander("Plate & Ship Setup", expanded=False):
        st.subheader("Plate")
        rows = st.number_input("Rows", min_value=1, value=int(config["plate_schema"].get("rows", 8)), step=1)
        cols = st.number_input("Columns", min_value=1, value=int(config["plate_schema"].get("columns", 12)), step=1)

        st.subheader("Ships")

        ship_types = list(ship_schema.keys())
        ships_to_delete = []

        # Display each ship in its own row with columns for length, count, and delete
        for ship in ship_types:
            disp_cols = st.columns([2, 2, 1, 1])
            with disp_cols[0]:
                st.markdown(f"**{ship}**")
            with disp_cols[1]:
                length = st.number_input(f"{ship} Length", min_value=1, value=ship_schema[ship]["length"], step=1, key=f"length_{ship}")
            with disp_cols[2]:
                count = st.number_input(f"{ship} Count", min_value=1, value=ship_schema[ship]["count"], step=1, key=f"count_{ship}")
            with disp_cols[3]:
                if st.button(f"🗑️", key=f"delete_{ship}"):
                    ships_to_delete.append(ship)
            ship_schema[ship] = {"length": length, "count": count}

        for ship in ships_to_delete:
            del ship_schema[ship]
            config["ship_schema"] = ship_schema
            save_config(config)
            st.rerun()  # Refresh the page to update UI

        # Add new ship section (not inside expander)
        st.markdown("---")
        st.markdown("**Add New Ship**")
        new_ship = st.text_input("New Ship Name", "")
        if new_ship and new_ship not in ship_types:
            new_length = st.number_input(f"{new_ship} Length", min_value=1, value=3, step=1, key=f"length_{new_ship}")
            new_count = st.number_input(f"{new_ship} Count", min_value=1, value=1, step=1, key=f"count_{new_ship}")
            if st.button("Add Ship", key="add_ship"):
                ship_schema[new_ship] = {"length": new_length, "count": new_count}
                config["ship_schema"] = ship_schema 
                save_config(config)
                st.success(f"Added {new_ship} with length {new_length} and count {new_count}")
                st.rerun()  #

        st.write("Current Ship Configuration:")
        st.json(ship_schema)

        if st.button("Save Configuration"):
            config["plate_schema"] = {"rows": int(rows), "columns": int(cols)}
            config["ship_schema"] = ship_schema  # Update with ship data from UI
            print("Saving configuration:", config)
            save_config(config)
            st.success("Configuration saved!")
            st.rerun()

    st.header("Plate Setup")
    placement_classes = find_placement_ai_classes()
    placement_names = list(placement_classes.keys())
    game_ai_classes = find_ai_classes()
    game_ai_names = list(game_ai_classes.keys())

    for plate_id in [1, 2]:
        st.subheader(f"Plate {plate_id}")
        place_choice = st.selectbox("Select Placement AI", options=placement_names, key=f"place_ai_{plate_id}")
        if st.button("Run placement AI", key=f"run_place_{plate_id}"):
            ai_cls = placement_classes[place_choice]
            placement = None
            for _ in range(5):
                try:
                    cand = ai_cls(game_shape, ship_schema).generate_placement()
                except Exception:
                    cand = None
                if cand and validate_placement_schema(cand, game_shape, ship_schema):
                    placement = cand
                    break
            if placement is None:
                placement = NaivePlacementAI(game_shape, ship_schema).generate_placement()
            st.session_state.placement[plate_id] = placement
        if st.session_state.placement[plate_id] is not None:
            st.pyplot(plot_ship_placement(game_shape, st.session_state.placement[plate_id]))
            if st.button("Rerun placement AI", key=f"rerun_{plate_id}"):
                st.session_state.placement[plate_id] = None
            if not st.session_state.liquids_placed[plate_id]:
                if st.button("Confirm and place liquids", key=f"confirm_{plate_id}"):
                    placement = st.session_state.placement[plate_id]
                    ship_cells = coords_from_schema(placement)
                    all_cells = [(r, c) for r in range(game_shape[0]) for c in range(game_shape[1])]
                    ocean_cells = [c for c in all_cells if c not in ship_cells]
                    ship_wells = [f"{ascii_uppercase[r]}{c+1}" for r, c in ship_cells]
                    ocean_wells = [f"{ascii_uppercase[r]}{c+1}" for r, c in ocean_cells]

                    calib_miss = [f"{ascii_uppercase[r]}{plate_shape_full[1]}" for r in range(4)]
                    calib_hit = [f"{ascii_uppercase[r]}{plate_shape_full[1]}" for r in range(4, 8)]

                    st.session_state.robot.add_place_water_action(plate_id, ocean_wells + calib_miss)
                    st.session_state.robot.add_place_ships_action(plate_id, ship_wells + calib_hit)
                    for well in calib_miss + calib_hit:
                        st.session_state.robot.add_fire_missile_action(plate_id, well)

                    st.session_state.robot.execute_actions_on_remote()
                    st.session_state.liquids_placed[plate_id] = True
        st.session_state.game_ai_choice[plate_id] = st.selectbox("Select gameplay AI", options=game_ai_names, key=f"game_ai_{plate_id}")

    ready = all(st.session_state.liquids_placed.values()) and all(st.session_state.game_ai_choice.values())
    start_button = st.button("🚀 Start Game", type="primary", use_container_width=True, disabled=not ready)

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
    Player1AI = game_ai_classes[st.session_state.game_ai_choice[1]]
    Player2AI = game_ai_classes[st.session_state.game_ai_choice[2]]
    p1_ai_choice = st.session_state.game_ai_choice[1]
    p2_ai_choice = st.session_state.game_ai_choice[2]

    player_1 = Player1AI("player_1", game_shape, ship_schema)
    player_2 = Player2AI("player_2", game_shape, ship_schema)

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
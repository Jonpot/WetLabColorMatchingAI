import json
import os
import subprocess
import tempfile
from typing import Tuple, Dict, Any, Optional
import tkinter as tk
from tkinter import filedialog

from battleship.ai.base_ai import BattleshipAI
from battleship.plate_state_processor import WellState


class GoWrapperAI(BattleshipAI):
    """A Battleship AI that delegates move selection to a Go executable."""

    def __init__(
        self,
        player_id: str,
        board_shape: Tuple[int, int],
        ship_schema: Dict[str, Any],
        go_executable: Optional[str] = None,
    ) -> None:
        super().__init__(player_id, board_shape, ship_schema)
        if go_executable is None:
            # Open a file dialog to select the Go executable
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            go_executable = filedialog.askopenfilename(title="Select Go Executable")
            root.destroy()
        self.go_executable = go_executable

    def select_next_move(self) -> Tuple[int, int]:
        board = [[cell.value for cell in row] for row in self.board_state]
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(board, tmp)
            tmp_path = tmp.name
        try:
            result = subprocess.run([self.go_executable, tmp_path], capture_output=True, text=True, check=True)
            output = result.stdout.strip().split()
            if len(output) != 2:
                raise ValueError(f"Invalid output from Go AI: {result.stdout}")
            return int(output[0]), int(output[1])
        finally:
            os.remove(tmp_path)
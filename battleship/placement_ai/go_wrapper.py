import json
import os
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog

from .base_placement_ai import PlacementAI


class GoPlacementWrapperAI(PlacementAI):
    """Placement AI that delegates placement generation to a Go executable."""

    def __init__(
        self,
        board_shape: Tuple[int, int],
        ship_schema: Dict[str, Any],
        go_executable: Optional[str] = None,
    ) -> None:
        super().__init__(board_shape, ship_schema)
        if go_executable is None:
            root = tk.Tk()
            root.withdraw()
            go_executable = filedialog.askopenfilename(title="Select Go Placement Executable")
            root.destroy()
        self.go_executable = go_executable

    def generate_placement(self) -> List[Dict[str, Any]]:
        data = {"board_shape": self.board_shape, "ship_schema": self.ship_schema}
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        try:
            result = subprocess.run([self.go_executable, tmp_path], capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        finally:
            os.remove(tmp_path)

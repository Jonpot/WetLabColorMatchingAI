"""
AI for Battleship Game

Will take as input the board state and return the best next move.
"""


from pathlib import Path
import sys 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Tuple
from string import ascii_uppercase
import numpy as np

from battleship.plate_state_processor import PlateStateProcessor, WellState
from battleship.robot.ot2_utils import OT2Manager


class AI:
    """AI for the Battleship game that determines the next move based on the plate state."""

    def __init__(self, plate_processor: PlateStateProcessor, robot: OT2Manager, ship_schema: Dict[str, Any]) -> None:
        """Initialize the AI with a plate schema and camera index."""
        self.plate_processor = plate_processor
        self.robot = robot
        self.ship_schema = ship_schema

        num_rows = plate_processor.plate_schema['rows']
        num_columns = plate_processor.plate_schema['columns']
        if not isinstance(num_rows, int) or not isinstance(num_columns, int):
            raise ValueError("Plate schema must contain 'rows' and 'columns' as integers.")

        self.board_state: np.ndarray = np.full((num_rows, num_columns), WellState.UNKNOWN, dtype=WellState)
        self.board_history: List[np.ndarray] = []
        self.board_history.append(self.board_state.copy())

    def main_loop(self) -> None:
        """Main loop for the AI to run until the game is over."""
        num_steps = 0
        while not self.determine_game_state():
            num_steps += 1
            next_move = self.get_next_move()
            if next_move:
                self.fire_missile(next_move)
                self.board_history.append(self.board_state.copy())
                print(f"Step {num_steps}: Fired at {ascii_uppercase[next_move[0]]}{next_move[1] + 1}.")
                print(f"It was a {self.board_state[next_move[0], next_move[1]]}!")
            else:
                print("No valid moves available. Exiting.")
                break
        print(f"Game Over! All ships sunk after {num_steps} steps.")

    def get_next_move(self) -> Tuple[int, int]:
        """Determine the next well to target based on the current board state.

        The algorithm uses a simple "hunt and target" strategy.  If there are
        any hits on the board, the AI prioritizes firing at neighbouring wells
        to try and finish that ship.  Otherwise it falls back to a probability
        density calculation which considers every possible placement of the
        remaining ships and fires at the well that is covered by the most valid
        placements.  When multiple wells have the same score one is chosen
        randomly.

        Returns
        -------
        Tuple[int, int]
            The row and column index of the next well to fire upon.
        """

        # Step 1: try to target around existing hits
        hit_clusters = self._get_hit_clusters()
        candidate_targets: List[Tuple[int, int]] = []
        for cluster in hit_clusters:
            candidate_targets.extend(self._get_cluster_targets(cluster))

        probability_map = self._calculate_probability_map()

        if candidate_targets:
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for pos in candidate_targets:
                if pos not in seen:
                    seen.add(pos)
                    unique_candidates.append(pos)
            # Choose the candidate with the highest probability score
            unique_candidates.sort(key=lambda p: probability_map[p[0], p[1]],
                                   reverse=True)
            return unique_candidates[0]

        # Step 2: hunt mode - choose globally highest probability
        if np.all(self.board_state != WellState.UNKNOWN):
            # No moves left
            raise RuntimeError("No valid moves remaining")

        max_prob = probability_map.max()
        if max_prob <= 0:
            unknown_cells = np.argwhere(self.board_state == WellState.UNKNOWN)
            choice = unknown_cells[0]
            return int(choice[0]), int(choice[1])

        max_indices = np.argwhere(probability_map == max_prob)
        choice = max_indices[np.random.choice(len(max_indices))]
        return int(choice[0]), int(choice[1])

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_hit_clusters(self) -> List[List[Tuple[int, int]]]:
        """Return lists of contiguous HIT coordinates."""
        hits = self.board_state == WellState.HIT
        visited = np.zeros_like(hits, dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []
        rows, cols = self.board_state.shape

        for i in range(rows):
            for j in range(cols):
                if hits[i, j] and not visited[i, j]:
                    stack = [(i, j)]
                    visited[i, j] = True
                    cluster: List[Tuple[int, int]] = []
                    while stack:
                        ci, cj = stack.pop()
                        cluster.append((ci, cj))
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < rows and 0 <= nj < cols and hits[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                stack.append((ni, nj))
                    clusters.append(cluster)
        return clusters

    def _get_cluster_targets(self, cluster: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Return potential target cells adjacent to a cluster of hits."""
        rows, cols = self.board_state.shape
        targets: List[Tuple[int, int]] = []

        if len(cluster) == 1:
            i, j = cluster[0]
            neighbours = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        else:
            rows_set = {c[0] for c in cluster}
            cols_set = {c[1] for c in cluster}
            if len(rows_set) == 1:  # horizontal orientation
                r = next(iter(rows_set))
                min_c = min(cols_set)
                max_c = max(cols_set)
                neighbours = [(r, min_c - 1), (r, max_c + 1)]
            elif len(cols_set) == 1:  # vertical orientation
                c = next(iter(cols_set))
                min_r = min(rows_set)
                max_r = max(rows_set)
                neighbours = [(min_r - 1, c), (max_r + 1, c)]
            else:
                neighbours = []
                for i, j in cluster:
                    neighbours.extend([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])

        for ni, nj in neighbours:
            if 0 <= ni < rows and 0 <= nj < cols and self.board_state[ni, nj] == WellState.UNKNOWN:
                targets.append((ni, nj))
        return targets

    def _calculate_probability_map(self) -> np.ndarray:
        """Calculate a heat map of ship placement probabilities."""
        rows, cols = self.board_state.shape
        probability_map = np.zeros((rows, cols), dtype=float)

        ship_lengths: List[int] = []
        for ship in self.ship_schema.values():
            ship_lengths.extend([ship["length"]] * int(ship["count"]))

        for length in ship_lengths:
            # Horizontal placements
            for i in range(rows):
                for j in range(cols - length + 1):
                    segment = [(i, j + k) for k in range(length)]
                    if any(self.board_state[x, y] == WellState.MISS for x, y in segment):
                        continue
                    for x, y in segment:
                        if self.board_state[x, y] != WellState.MISS:
                            probability_map[x, y] += 1

            # Vertical placements
            for i in range(rows - length + 1):
                for j in range(cols):
                    segment = [(i + k, j) for k in range(length)]
                    if any(self.board_state[x, y] == WellState.MISS for x, y in segment):
                        continue
                    for x, y in segment:
                        if self.board_state[x, y] != WellState.MISS:
                            probability_map[x, y] += 1

        # Avoid suggesting already fired wells
        probability_map[self.board_state != WellState.UNKNOWN] = 0
        return probability_map

    def fire_missile(self, targeted_well: Tuple[int, int]) -> None:
        """Fire a missile at the targeted well and updates the board state."""
        well_name = f"{ascii_uppercase[targeted_well[0]]}{targeted_well[1] + 1}"
        self.robot.add_fire_missile_action(well_name)
        self.robot.execute_actions_on_remote()

        well_state = self.plate_processor.determine_well_state((targeted_well[0], targeted_well[1]))
        self.board_state[targeted_well[0], targeted_well[1]] = well_state

    def determine_game_state(self) -> bool:
        """Determine the current game state based on the board state.

        This should consider all the ships and their statuses based on the ship schema
        and the current board state.

        If all ships are sunk, the game is over.
        If any ship is still afloat, the game continues.

        This can be simplified to checking if there are as many hits as there are total ship segments.

        Returns:
            bool: True if the game is over (all ships sunk), False otherwise.
        """
        total_segments = sum(ship['length'] * ship['count'] for ship in self.ship_schema.values())
        hit_segments = np.sum(self.board_state == WellState.HIT)

        return hit_segments >= total_segments
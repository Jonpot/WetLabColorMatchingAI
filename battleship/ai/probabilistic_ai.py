import numpy as np
from typing import Tuple, List
from battleship.ai.base_ai import BattleshipAI
from battleship.plate_state_processor import WellState

class JonsProbabilisticAI(BattleshipAI):
    """
    A probabilistic "hunt and target" AI implementation for Battleship.
    This can be used as an example for students or as a default competitor.
    """
    def select_next_move(self) -> Tuple[int, int]:
        # Step 1: Target mode - fire adjacent to existing hits
        hit_clusters = self._get_hit_clusters()
        candidate_targets = []
        for cluster in hit_clusters:
            candidate_targets.extend(self._get_cluster_targets(cluster))

        probability_map = self._calculate_probability_map()

        if candidate_targets:
            # Sort unique candidates by their probability score
            unique_candidates = sorted(list(set(candidate_targets)),
                                       key=lambda p: probability_map[p],
                                       reverse=True)
            return unique_candidates[0]

        # Step 2: Hunt mode - fire at the globally highest probability cell
        max_prob = np.max(probability_map)
        if max_prob > 0:
            max_indices = np.argwhere(probability_map == max_prob)
            choice = max_indices[np.random.choice(len(max_indices))]
            return tuple(choice)
        
        # Fallback: if map is all zeros, pick any unknown cell
        unknown_cells = np.argwhere(self.board_state == WellState.UNKNOWN)
        if unknown_cells.size > 0:
            return tuple(unknown_cells[0])
        
        raise RuntimeError("No valid moves remaining.")

    # Helper methods for probability calculation (moved from original AI class)
    def _calculate_probability_map(self) -> np.ndarray:
        # This calculates a heat map of ship placement probabilities.
        rows, cols = self.board_shape
        prob_map = np.zeros((rows, cols), dtype=int)
        ship_lengths = [ship["length"] for ship in self.ship_schema.values() for _ in range(ship["count"])]

        for length in ship_lengths:
            for r in range(rows):
                for c in range(cols - length + 1): # Horizontal
                    if all(self.board_state[r, c+k] != WellState.MISS for k in range(length)):
                        for k in range(length): prob_map[r, c+k] += 1
            for r in range(rows - length + 1):
                for c in range(cols): # Vertical
                    if all(self.board_state[r+k, c] != WellState.MISS for k in range(length)):
                        for k in range(length): prob_map[r+k, c] += 1
        
        prob_map[self.board_state != WellState.UNKNOWN] = -1 # Exclude known cells
        return prob_map

    def _get_hit_clusters(self) -> List[List[Tuple[int, int]]]:
        # This finds groups of contiguous 'HIT' cells.
        hits = np.argwhere(self.board_state == WellState.HIT)
        if not hits.any(): return []
        
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=1.1, min_samples=1).fit(hits)
        
        clusters = []
        for label in np.unique(clustering.labels_):
            if label != -1:
                clusters.append([tuple(p) for p in hits[clustering.labels_ == label]])
        return clusters

    def _get_cluster_targets(self, cluster: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # This finds valid UNKNOWN cells adjacent to a cluster of hits.
        targets = set()
        rows, cols = self.board_shape

        for r, c in cluster:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and self.board_state[nr, nc] == WellState.UNKNOWN:
                    targets.add((nr, nc))
        return list(targets)
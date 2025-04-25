import numpy as np
import random
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls
from scipy.linalg import lstsq

class ColorLearningOptimizer:
    def __init__(self, 
                 dye_count: int,
                 max_well_volume: int = 200,
                 step: int = 1,
                 tolerance: int = 30,
                 min_required_volume: int = 20,
                 optimization_mode: str = "unmixing"  # or "random_forest"
                ):
        self.dye_count = dye_count
        self.max_well_volume = max_well_volume
        self.step = step
        self.tolerance = tolerance
        self.min_required_volume = min_required_volume
        self.optimization_mode = optimization_mode
        self.X_train = []
        self.Y_train = []

    def reset(self):
        self.X_train = []
        self.Y_train = []

    def add_data(self, volumes: list, measured_color: list):
        self.X_train.append(volumes)
        self.Y_train.append(measured_color)

    def suggest_next_experiment(self, target_color: list) -> list:
        if self.optimization_mode == "unmixing":
            volumes = self._color_unmixing_optimize(target_color)
        elif self.optimization_mode == "random_forest":
            volumes = self._random_forest_optimize(target_color)
        else:
            raise ValueError(f"Unknown optimization mode: {self.optimization_mode}")
        return self._apply_min_volume_constraint(volumes)
    
    
    def calculate_distance(self, color: list, target_color: list) -> float:
        print(f"Running calculate distance. Color: {color}, type: {type(color)}, target_color: {target_color}, type: {type(target_color)}")
        print(f"Sum: {sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color, target_color))}")
        print(f"Raw sqrt: {math.sqrt(sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color, target_color)))}")
        return math.sqrt(sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color, target_color)))

    def within_tolerance(self, color: list, target_color: list) -> bool:
        return self.calculate_distance(color, target_color) <= self.tolerance

    def _random_combination(self) -> list:
        vols = [0] * self.dye_count
        remain = self.max_well_volume

        if remain > 0 and self.dye_count > 0:
            primary_dye = random.randint(0, self.dye_count - 1)
            vols[primary_dye] = self.step
            remain -= self.step

        for i in range(self.dye_count):
            if remain <= 0:
                break
            if random.random() < 0.7:
                possible_steps = remain // self.step
                if possible_steps > 0:
                    vol = random.randint(0, possible_steps) * self.step
                    vols[i] += vol
                    remain -= vol

        if remain > 0 and self.dye_count > 0:
            lucky_dye = random.randint(0, self.dye_count - 1)
            vols[lucky_dye] += remain

        return vols

    def _apply_min_volume_constraint(self, volumes: list) -> list:
        # Remove dyes with volume below threshold
        volumes = [v if v >= self.min_required_volume else 0 for v in volumes]

        total_vol = sum(volumes)
        if total_vol == 0:
            return self._random_combination()

        # Rescale to total max_well_volume
        scale = self.max_well_volume / total_vol
        volumes = [int(v * scale) for v in volumes]

        diff = self.max_well_volume - sum(volumes)
        if diff != 0:
            max_idx = np.argmax(volumes)
            volumes[max_idx] += diff

        return volumes

    def _random_forest_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) == 0:
            random_volumes = self._random_combination()
            return self._apply_min_volume_constraint(random_volumes)

        scores = np.array([self._color_distance_score(color, target_rgb) for color in self.Y_train])
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        n_estimators = min(100, max(10, len(self.X_train) * 5))
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        try:
            rf.fit(X_train_scaled, scores)
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            return self._random_combination()

        candidates = [self._random_combination() for _ in range(50)]
        X_candidates = np.array(candidates)
        X_candidates_scaled = scaler.transform(X_candidates)

        try:
            predicted_scores = rf.predict(X_candidates_scaled)
            tree_predictions = np.array([tree.predict(X_candidates_scaled) for tree in rf.estimators_])
            uncertainties = np.var(tree_predictions, axis=0)
            acquisition = predicted_scores + uncertainties
            sorted_indices = np.argsort(acquisition)[::-1]
            best = candidates[sorted_indices[0]]
            return self._apply_min_volume_constraint(best)

        except Exception as e:
            print(f"Random Forest prediction failed: {e}")
            return self._random_combination()

    def _color_distance_score(self, color, target_rgb, max_distance=441.7):
        distance = self.calculate_distance(color, target_rgb)
        return (1.0 - distance / max_distance) ** 3

    def _color_unmixing_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) < 2:
            random_volumes = self._random_combination()
            return self._apply_min_volume_constraint(random_volumes)

        X = np.array(self.X_train)
        X_norm = np.array([row / (np.sum(row) if np.sum(row) > 0 else 1) for row in X])
        Y = np.array(self.Y_train)

        try:
            dye_colors = np.zeros((self.dye_count, 3))
            for i in range(3):
                result = lstsq(X_norm, Y[:, i], cond=None)
                dye_colors[:, i] = result[0]
            dye_colors = np.clip(dye_colors, 0, 255)

            weights, _ = nnls(dye_colors, np.array(target_rgb))
            if sum(weights) > 0:
                weights = weights / sum(weights)

            volumes = [int(w * self.max_well_volume) for w in weights]
            return self._apply_min_volume_constraint(volumes)

        except Exception as e:
            print(f"Color unmixing optimization failed: {e}")
            return self._random_combination()

import numpy as np
import random
import math
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
                 optimization_mode: str = "mlp_active",
                 n_models: int = 5,
                 exploration_weight: float = 1.0,
                 initial_explore_count: int = 0,
                 initial_force_all_dyes: bool = False,
                 candidate_num: int = 300,
                 single_row_learning: bool = True,):
        self.dye_count = dye_count
        self.max_well_volume = max_well_volume
        self.step = step
        self.tolerance = tolerance
        self.min_required_volume = min_required_volume
        self.optimization_mode = optimization_mode
        self.n_models = n_models
        self.exploration_weight = exploration_weight
        self.initial_explore_count = initial_explore_count
        self.initial_force_all_dyes = initial_force_all_dyes
        self.candidate_num = candidate_num
        self.single_row_learning = single_row_learning


        self.X_train = []
        self.Y_train = []

        if self.optimization_mode == "mlp_active":
            self.models = [MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, random_state=42+i) for i in range(self.n_models)]

    def _forced_full_dye_combination(self) -> list:
        vols = [self.min_required_volume] * self.dye_count
        total = sum(vols)
        scale = self.max_well_volume / total
        vols = [int(v * scale) for v in vols]

        diff = self.max_well_volume - sum(vols)
        if diff != 0:
            vols[np.argmax(vols)] += diff
        return vols
    
    def reset(self):
        if self.single_row_learning:
            self.X_train = []
            self.Y_train = []


    def add_data(self, volumes: list, measured_color: list):
        self.X_train.append(volumes)
        self.Y_train.append(measured_color)
        if self.optimization_mode == "mlp_active" and len(self.X_train) >= 2:
            for model in self.models:
                model.fit(self.X_train, self.Y_train)

    def suggest_next_experiment(self, target_color: list) -> list:
        if self.initial_force_all_dyes and self.initial_explore_count < 2:
            self.initial_explore_count += 1
            return self._forced_full_dye_combination()
        if self.optimization_mode == "mlp_active":
            volumes = self._mlp_active_optimize(target_color)
        elif self.optimization_mode == "unmixing":
            volumes = self._color_unmixing_optimize(target_color)
        elif self.optimization_mode == "random_forest":
            volumes = self._random_forest_optimize(target_color)
        elif self.optimization_mode == "extratree_active":
            volumes = self._extratree_active_optimize(target_color)
        else:
            raise ValueError(f"Unknown optimization mode: {self.optimization_mode}")
        return volumes

    def calculate_distance(self, color, target_color) -> float:
        return math.sqrt(
            sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color, target_color))
        )


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

        return self._apply_min_volume_constraint(vols)

    def _apply_min_volume_constraint(self, volumes: list) -> list:
        adjusted = []
        for v in volumes:
            if v == 0:
                adjusted.append(0)
            elif v < self.min_required_volume:
                if abs(v - 0) < abs(v - self.min_required_volume):
                    adjusted.append(0)
                else:
                    adjusted.append(self.min_required_volume)
            else:
                adjusted.append(v)

        total_vol = sum(adjusted)
        if total_vol == 0:
            # avoid infinite recursion when min_required_volume exceeds step
            idx = random.randrange(self.dye_count)
            adjusted[idx] = self.min_required_volume
            total_vol = sum(adjusted)

        scale = self.max_well_volume / total_vol

        # convert scaled volumes to integer multiples of the step size
        scaled = [v * scale for v in adjusted]
        units = [int(round(val / self.step)) for val in scaled]
        target_units = self.max_well_volume // self.step
        diff_units = target_units - sum(units)
        if diff_units != 0:
            max_idx = np.argmax(units)
            units[max_idx] += diff_units

        adjusted = [u * self.step for u in units]

        return adjusted

    def _mlp_active_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) < 2:
            return self._random_combination()

        candidates = [self._random_combination() for _ in range(self.candidate_num)]
        candidates_np = np.array(candidates)

        all_preds = np.array([model.predict(candidates_np) for model in self.models])
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        mean_distances = np.linalg.norm(mean_preds - np.array(target_rgb), axis=1)
        uncertainty_scores = np.linalg.norm(std_preds, axis=1)

        acquisition_scores = -mean_distances + self.exploration_weight * uncertainty_scores

        best_idx = np.argmax(acquisition_scores)
        return candidates[best_idx]

    def _extratree_active_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) == 0:
            return self._random_combination()

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)

        n_estimators = min(100, max(10, len(self.X_train) * 5))
        et = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42)

        try:
            et.fit(X_train_scaled, self.Y_train)
        except Exception as e:
            print(f"ExtraTrees training failed: {e}")
            return self._random_combination()

        candidates = [self._random_combination() for _ in range(self.candidate_num)]
        X_candidates_scaled = scaler.transform(candidates)

        try:
            preds = np.array([tree.predict(X_candidates_scaled) for tree in et.estimators_])
            mean_preds = np.mean(preds, axis=0)
            std_preds = np.std(preds, axis=0)

            mean_distances = np.linalg.norm(mean_preds - np.array(target_rgb), axis=1)
            uncertainty_scores = np.linalg.norm(std_preds, axis=1)

            acquisition_scores = -mean_distances + self.exploration_weight * uncertainty_scores

            best_idx = np.argmax(acquisition_scores)
            return candidates[best_idx]

        except Exception as e:
            print(f"ExtraTrees prediction failed: {e}")
            return self._random_combination()

    def _random_forest_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) == 0:
            return self._random_combination()

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

        candidates = [self._random_combination() for _ in range(self.candidate_num)]
        X_candidates_scaled = scaler.transform(candidates)

        try:
            predicted_scores = rf.predict(X_candidates_scaled)
            tree_predictions = np.array([tree.predict(X_candidates_scaled) for tree in rf.estimators_])
            uncertainties = np.var(tree_predictions, axis=0)
            acquisition = predicted_scores + uncertainties
            sorted_indices = np.argsort(acquisition)[::-1]
            best = candidates[sorted_indices[0]]
            return best

        except Exception as e:
            print(f"Random Forest prediction failed: {e}")
            return self._random_combination()

    def _color_distance_score(self, color, target_rgb, max_distance=441.7):
        distance = self.calculate_distance(color, target_rgb)
        return (1.0 - distance / max_distance) ** 3

    def _color_unmixing_optimize(self, target_rgb: list) -> list:
        if len(self.X_train) < 2:
            return self._random_combination()

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

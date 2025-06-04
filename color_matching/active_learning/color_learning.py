"""Active learning optimizer used by the colour matching robot."""

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
                 n_models: int = 5,
                 exploration_weight: float = 1.0,
                 initial_explore_count: int = 3,
                 initial_force_all_dyes: bool = False,
                 single_row_learning: bool = True,):
        self.dye_count = dye_count
        self.max_well_volume = max_well_volume
        self.step = step
        self.tolerance = tolerance
        self.min_required_volume = min_required_volume
        self.n_models = n_models
        self.exploration_weight = exploration_weight
        self.initial_explore_count = initial_explore_count
        self.initial_force_all_dyes = initial_force_all_dyes
        self.single_row_learning = single_row_learning


        self.X_train = []
        self.Y_train = []

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
        if len(self.X_train) >= 2:
            for model in self.models:
                model.fit(self.X_train, self.Y_train)

    def suggest_next_experiment(self, target_color: list) -> list:
        if self.initial_force_all_dyes and self.initial_explore_count > 0:
            self.initial_explore_count -= 1
            return self._forced_full_dye_combination()
        volumes = self._mlp_active_optimize(target_color)
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
        """Suggest volumes by directly optimising the model prediction."""
        if len(self.X_train) < 2:
            return self._random_combination()

        from scipy.optimize import minimize

        target = np.array(target_rgb)

        def objective(vols: np.ndarray) -> float:
            vols = np.clip(vols, 0, self.max_well_volume)
            vols = self._apply_min_volume_constraint(vols.tolist())
            x = np.array(vols).reshape(1, -1)
            preds = np.array([m.predict(x)[0] for m in self.models])
            mean_pred = np.mean(preds, axis=0)
            std_pred = np.std(preds, axis=0)
            dist = np.linalg.norm(mean_pred - target)
            dist -= self.exploration_weight * np.linalg.norm(std_pred)
            return dist

        x0 = np.array(self._random_combination(), dtype=float)
        bounds = [(0, self.max_well_volume) for _ in range(self.dye_count)]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})
        best_vols = result.x if result.success else x0
        return self._apply_min_volume_constraint(best_vols.tolist())
"""Active learning optimizer used by the colour matching robot."""

"""Active learning utilities for colour matching."""

import math
import random
from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class ColorLearningOptimizer:
    """Bayesian optimisation based colour mixing helper."""

    def __init__(self,
                 dye_count: int,
                 max_well_volume: int = 200,
                 step: int = 1,
                 tolerance: int = 30,
                 min_required_volume: int = 20,
                 n_models: int = 3,
                 exploration_weight: float = 1.0,
                 single_row_learning: bool = True,
                 optimization_mode: str | None = None,
                 **_ignored: object) -> None:
        self.dye_count = dye_count
        self.max_well_volume = max_well_volume
        self.step = step
        self.tolerance = tolerance
        self.min_required_volume = min_required_volume
        self.exploration_weight = exploration_weight
        self.single_row_learning = single_row_learning

        # Gaussian processes for each RGB channel
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel()
        self.models: List[GaussianProcessRegressor] = [
            GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42 + i)
            for i in range(3)
        ]

        print("Initialized ColorLearningOptimizer with parameters:")
        print(f"  Dye count: {self.dye_count}")
        print(f"  Max well volume: {self.max_well_volume}")
        print(f"  Step size: {self.step}")
        print(f"  Tolerance: {self.tolerance}")
        print(f"  Min required volume: {self.min_required_volume}")
        self.X_train: List[List[int]] = []
        self.Y_train: List[List[int]] = []
    
    def reset(self):
        if self.single_row_learning:
            print("Resetting optimizer for single row learning.")
            self.X_train: List[List[int]] = []
            self.Y_train: List[List[int]] = []


    def add_data(self, volumes: list, measured_color: list) -> None:
        """Add an observed colour measurement."""
        self.X_train.append(volumes)
        self.Y_train.append(measured_color)
        if len(self.X_train) >= 1:
            X = np.array(self.X_train)
            Y = np.array(self.Y_train)
            for c, model in enumerate(self.models):
                model.fit(X, Y[:, c])
            print(f"Trained models with {len(self.X_train)} samples.")
            print(f"Current training data: {self.X_train} -> {self.Y_train}")

    def suggest_next_experiment(self, target_color: list) -> list:
        """Propose the next dye volumes to test."""
        volumes = self._gp_optimize(target_color)
        print(f"Suggesting volumes: {volumes} for target color: {target_color}")
        return volumes

    def calculate_distance(self, color, target_color) -> float:
        return math.sqrt(
            sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color, target_color))
        )

    def update_exploration_weight(self, new_weight: float) -> None:
        """Update the exploration weight for the optimization."""
        self.exploration_weight = new_weight
        print(f"Updated exploration weight to: {self.exploration_weight}")

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

    def _gp_optimize(self, target_rgb: list) -> list:
        """Suggest volumes by optimising the Gaussian process model."""
        if len(self.X_train) < self.dye_count + 1:
            return self._random_combination()

        from scipy.optimize import minimize

        target = np.array(target_rgb)

        def objective(vols: np.ndarray, report = False) -> float:
            vols = np.clip(vols, 0, self.max_well_volume)
            vols = self._apply_min_volume_constraint(vols.tolist())
            x = np.array(vols).reshape(1, -1)

            means = []
            stds = []
            for model in self.models:
                mean, std = model.predict(x, return_std=True)
                means.append(mean[0])
                stds.append(std[0])

            mean_pred = np.array(means)
            std_pred = np.array(stds)
            dist = np.linalg.norm(mean_pred - target)
            if report:
                print(f"Objective: {dist}, Means: {mean_pred}, Stds: {std_pred}")
                if dist < self.exploration_weight * np.linalg.norm(std_pred):
                    print("These volumes were chosen primarily based on exploitation.")
                else:
                    print("These volumes were chosen primarily based on exploration.")
            return dist - self.exploration_weight * np.linalg.norm(std_pred)

        x0 = np.array(self._random_combination(), dtype=float)
        bounds = [(0, self.max_well_volume) for _ in range(self.dye_count)]

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )

        best_vols = result.x if result.success else x0
        return self._apply_min_volume_constraint(best_vols.tolist())

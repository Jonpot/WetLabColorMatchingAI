"""Active learning optimizer used by the colour matching robot."""

"""Active learning utilities for colour matching."""

import math
import random
from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import LinearConstraint

class ColorLearningOptimizer:
    """Bayesian optimisation based colour mixing helper."""

    def __init__(self,
                 dye_count: int,
                 max_well_volume: int = 200,
                 step: int = 20,
                 tolerance: int = 30,
                 min_required_volume: int = 20,
                 n_models: int = 3,
                 exploration_weight: float = 0.4,
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
        self.major_unseen_dyes = set()
        for i in range(dye_count):
            self.major_unseen_dyes.add(i)

        # Gaussian processes for each RGB channel
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3)) + WhiteKernel()
        #kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3))
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
        print("Resetting optimizer for single row learning.")
        self.X_train: List[List[int]] = []
        self.Y_train: List[List[int]] = []

    def add_data(self, volumes: list, measured_color: list) -> None:
        """Add an observed colour measurement."""
        self.X_train.append(volumes)
        self.Y_train.append(measured_color)
        self.train()

    def train(self) -> None:
        """
        Sometimes, it's necessary to just retrain the model instead of adding
        more data.
        """
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
        remaining_volume = self.max_well_volume

        if remaining_volume > 0 and self.dye_count > 0:
            if len(self.major_unseen_dyes) > 0:
                # Use a major unseen dye if available
                primary_dye = random.choice(list(self.major_unseen_dyes))
                self.major_unseen_dyes.remove(primary_dye)
            else:
                primary_dye = random.randint(0, self.dye_count - 1)
            possible_steps = remaining_volume // self.step
            vols[primary_dye] = random.randint(possible_steps//2, possible_steps) * self.step
            remaining_volume -= vols[primary_dye]

        for i in range(self.dye_count):
            if remaining_volume <= 0:
                break
            if random.random() < 0.7:
                possible_steps = remaining_volume // self.step
                if possible_steps > 0:
                    vol = random.randint(0, possible_steps) * self.step
                    vols[i] += vol
                    remaining_volume -= vol

        if remaining_volume > 0 and self.dye_count > 0:
            lucky_dye = random.randint(0, self.dye_count - 1)
            vols[lucky_dye] += remaining_volume

        return self._apply_min_volume_constraint(vols)

    def _apply_min_volume_constraint(self, volumes: list) -> list:
        """
        Applies two constraints to a list of continuous volumes:
        1. Any volume > 0 must be >= min_required_volume.
        2. The final integer volumes must sum to max_well_volume.
        """
        # 1. Apply the 'v == 0 or v >= min_required_volume' constraint
        adjusted = []
        for v in volumes:
            if self.min_required_volume/2 < v < self.min_required_volume:
                # Snap small values to the minimum required volume
                adjusted.append(self.min_required_volume)
            elif 0 < v < self.min_required_volume/2:
                adjusted.append(0)
            else:
                adjusted.append(v)

        # 2. Normalize the volumes to sum to max_well_volume
        total_vol = sum(adjusted)
        if total_vol == 0:
            # Handle the edge case where all inputs were zero
            idx = random.randrange(self.dye_count)
            adjusted[idx] = self.max_well_volume
            return [int(v) for v in adjusted]

        scale = self.max_well_volume / total_vol
        scaled_volumes = [v * scale for v in adjusted]

        # 3. Convert to integers and correct for rounding errors
        final_volumes = [int(round(v)) for v in scaled_volumes]
        
        # Correct sum due to rounding
        diff = self.max_well_volume - sum(final_volumes)
        if diff != 0:
            # Add the difference to the largest volume to minimize relative error
            max_idx = np.argmax(final_volumes)
            final_volumes[max_idx] += diff

        return final_volumes

    def _gp_optimize(self, target_rgb: list) -> list:
        """Suggest volumes by optimizing the Gaussian process model with multiple random restarts."""
        if len(self.X_train) < self.dye_count + 1:
            return self._random_combination()

        from scipy.optimize import minimize

        target = np.array(target_rgb)

        def objective(vols: np.ndarray, report: bool = False) -> float:
            vols = np.clip(vols, 0, self.max_well_volume)
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
                print(f"Literal comparison: {dist} vs {(np.linalg.norm(std_pred)/(self.exploration_weight+1e-6))} ({np.linalg.norm(std_pred)}/{(self.exploration_weight+1e-6)})")
                if dist < (np.linalg.norm(std_pred)/(self.exploration_weight+1e-6)):
                    print("These volumes were chosen primarily based on exploitation.")
                else:
                    print("These volumes were chosen primarily based on exploration.")

            return dist - self.exploration_weight * np.linalg.norm(std_pred)

        # Define the constraint: d1 + d2 + d3 = 200
        # The vector [1, 1, 1] means we sum all dye volumes.
        # The lower and upper bounds are both 200, forcing the sum to be exactly 200.
        constraint = LinearConstraint(np.ones(self.dye_count), [self.max_well_volume], [self.max_well_volume])


        bounds = [(0, self.max_well_volume) for _ in range(self.dye_count)]
        num_restarts = 30

        best_val = float('inf')
        best_vols_continuous = None

        for _ in range(num_restarts):
            x0_rand = np.random.rand(self.dye_count)
            x0 = (x0_rand / np.sum(x0_rand)) * self.max_well_volume
            result = minimize(
                objective,
                x0,
                method="trust-constr",
                bounds=bounds,
                options={"maxiter": 30},
                constraints=[constraint],
            )

            if result.success:
                candidate = result.x
                candidate_val = objective(candidate)
            else:
                candidate = x0
                candidate_val = objective(x0)

            if candidate_val < best_val:
                best_val = candidate_val
                best_vols_continuous = candidate

        # Report the final chosen volumes
        objective(best_vols_continuous, report=True)

        # Convert the continuous optimum to discrete, valid volumes
        best_vols_continuous = result.x
        final_discrete_vols = self._apply_min_volume_constraint(best_vols_continuous.tolist())

        return final_discrete_vols

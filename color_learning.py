import numpy as np
import math
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def calculate_distance_to_target(color, target_rgb):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, target_rgb)))

def color_distance_score(color, target_rgb, max_distance=441.7):
    distance = calculate_distance_to_target(color, target_rgb)
    score = 1.0 - (distance / max_distance)
    return score ** 3

def within_tolerance(color, target_rgb, tolerance):
    distance = calculate_distance_to_target(color, target_rgb)
    return distance <= tolerance

def random_combination(dye_count, step, max_volume):
    vols = [0] * dye_count
    remain = max_volume

    if remain > 0 and dye_count > 0:
        primary_dye = random.randint(0, dye_count-1)
        min_vol = step
        vols[primary_dye] = min_vol
        remain -= min_vol

    for i in range(dye_count):
        if remain <= 0:
            break
        if random.random() < 0.7:
            possible_steps = remain // step
            if possible_steps > 0:
                vol = random.randint(0, possible_steps) * step
                vols[i] += vol
                remain -= vol

    if remain > 0 and dye_count > 0:
        lucky_dye = random.randint(0, dye_count-1)
        vols[lucky_dye] += remain

    return vols

def generate_diverse_candidates(dye_count, n_samples, existing_experiments, max_volume, step):
    candidates = []
    seen = set(tuple(exp) for exp in existing_experiments or [])
    n_to_generate = n_samples

    for i in range(dye_count):
        candidate = [0] * dye_count
        candidate[i] = max_volume
        tuple_candidate = tuple(candidate)
        if tuple_candidate not in seen:
            seen.add(tuple_candidate)
            candidates.append(candidate)
            n_to_generate -= 1

    if dye_count >= 2:
        for i in range(dye_count):
            for j in range(i+1, dye_count):
                for ratio in [0.2, 0.5, 0.8]:
                    candidate = [0] * dye_count
                    candidate[i] = int(max_volume * ratio / step) * step
                    candidate[j] = max_volume - candidate[i]
                    tuple_candidate = tuple(candidate)
                    if tuple_candidate not in seen:
                        seen.add(tuple_candidate)
                        candidates.append(candidate)
                        n_to_generate -= 1
                        if n_to_generate <= 0:
                            return candidates

    if dye_count >= 3:
        for i in range(dye_count):
            for j in range(i+1, dye_count):
                for k in range(j+1, dye_count):
                    candidate = [0] * dye_count
                    candidate[i] = int(max_volume / 3)
                    candidate[j] = int(max_volume / 3)
                    candidate[k] = max_volume - candidate[i] - candidate[j]
                    tuple_candidate = tuple(candidate)
                    if tuple_candidate not in seen:
                        seen.add(tuple_candidate)
                        candidates.append(candidate)
                        n_to_generate -= 1
                        if n_to_generate <= 0:
                            return candidates

    attempts = 0
    while n_to_generate > 0 and attempts < n_to_generate * 5:
        attempts += 1
        candidate = random_combination(dye_count, step, max_volume)
        tuple_candidate = tuple(candidate)
        if tuple_candidate not in seen:
            seen.add(tuple_candidate)
            candidates.append(candidate)
            n_to_generate -= 1

    return candidates

def generate_diverse_covering_combinations(dye_count, n_combinations, max_volume, step):
    combinations = []
    used_dyes = set()
    attempts = 0
    max_attempts = 1000

    while len(combinations) < n_combinations and attempts < max_attempts:
        attempts += 1
        candidate = random_combination(dye_count, step, max_volume)
        nonzero_indices = [i for i, v in enumerate(candidate) if v > 0]

        if len(nonzero_indices) < 2:
            continue

        new_dyes = set(nonzero_indices) - used_dyes
        if new_dyes or len(combinations) == 0:
            if not any(np.array_equal(candidate, c) for c in combinations):
                combinations.append(candidate)
                used_dyes.update(nonzero_indices)

        if len(used_dyes) == dye_count:
            break

    while len(combinations) < n_combinations:
        candidate = random_combination(dye_count, step, max_volume)
        if len([v for v in candidate if v > 0]) < 2:
            continue
        if not any(np.array_equal(candidate, c) for c in combinations):
            combinations.append(candidate)

    return combinations

def random_forest_optimize_next_experiment(X_train, Y_train, target_rgb, dye_count, max_volume, step, max_iterations=11):
    if len(X_train) == 0:
        return random_combination(dye_count, step, max_volume)

    scores = np.array([color_distance_score(color, target_rgb) for color in Y_train])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    n_estimators = min(100, max(10, len(X_train) * 5))

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    try:
        rf.fit(X_train_scaled, scores)
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        return random_combination(dye_count, step, max_volume)

    candidates = generate_diverse_candidates(dye_count, 50, X_train, max_volume, step)
    if not candidates:
        print("No unique candidates found, falling back to random")
        while True:
            new_candidate = random_combination(dye_count, step, max_volume)
            if not any(np.array_equal(new_candidate, exp) for exp in X_train):
                return new_candidate

    X_candidates = np.array(candidates)
    X_candidates_scaled = scaler.transform(X_candidates)
    try:
        predicted_scores = rf.predict(X_candidates_scaled)
        tree_predictions = np.array([tree.predict(X_candidates_scaled) for tree in rf.estimators_])
        uncertainties = np.var(tree_predictions, axis=0)

        ratio = len(X_train) / max_iterations
        exploration_weight = 2.0 if ratio < 0.3 else 1.0 if ratio < 0.7 else 0.5

        acquisition = predicted_scores + exploration_weight * uncertainties

        if len(X_train) >= 3:
            best_indices = np.argsort(scores)[-3:]
            best_X = X_train_scaled[best_indices]
            for best_x in best_X:
                distances = np.sqrt(np.sum((X_candidates_scaled - best_x) ** 2, axis=1))
                proximity_bonus = 0.3 * np.exp(-distances)
                acquisition += proximity_bonus

        sorted_indices = np.argsort(acquisition)[::-1]
        return candidates[sorted_indices[0]]

    except Exception as e:
        print(f"Random Forest prediction failed: {e}")
        return random_combination(dye_count, step, max_volume)

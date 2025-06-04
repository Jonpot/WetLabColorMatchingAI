import importlib.util
import unittest

SKIP = importlib.util.find_spec("numpy") is None or importlib.util.find_spec("sklearn") is None

if not SKIP:
    import numpy as np
    from active_learning.color_learning import ColorLearningOptimizer


@unittest.skipIf(SKIP, "numpy and sklearn are required")
class ColorLearningTests(unittest.TestCase):
    def test_apply_min_volume_constraint(self):
        opt = ColorLearningOptimizer(dye_count=3, max_well_volume=100, min_required_volume=20)
        volumes = [10, 30, 50]
        result = opt._apply_min_volume_constraint(volumes)
        self.assertEqual(sum(result), 100)
        for v in result:
            if v != 0:
                self.assertGreaterEqual(v, opt.min_required_volume)
            self.assertEqual(v % opt.step, 0)

    def test_random_combination_respects_constraints(self):
        opt = ColorLearningOptimizer(dye_count=3, max_well_volume=200, step=10)
        vols = opt._random_combination()
        self.assertEqual(sum(vols), 200)
        for v in vols:
            self.assertEqual(v % opt.step, 0)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, opt.max_well_volume)

    def test_active_training_and_suggestion(self):
        opt = ColorLearningOptimizer(
            dye_count=2,
            max_well_volume=100,
            step=10,
        )
        # simple linear mixing model: dye1 controls red, dye2 controls green
        for v1 in range(0, 101, 50):
            v2 = 100 - v1
            color = [int(255 * v1 / 100), int(255 * v2 / 100), 0]
            opt.add_data([v1, v2], color)

        target = [128, 128, 0]
        suggestion = opt.suggest_next_experiment(target)
        self.assertEqual(len(suggestion), 2)
        self.assertEqual(sum(suggestion), opt.max_well_volume)


if __name__ == "__main__":
    unittest.main()

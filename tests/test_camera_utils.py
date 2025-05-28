import importlib.util
import unittest

SKIP = importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None

if not SKIP:
    import numpy as np
    import cv2
    import os
    import json
    import tempfile
    from unittest.mock import patch
    from camera.camera_w_calibration import PlateProcessor, srgb2lin, lin2srgb


@unittest.skipIf(SKIP, "numpy and cv2 are required")
class CameraUtilsTests(unittest.TestCase):
    def test_well_centers_shape(self):
        centers = PlateProcessor.well_centers(0, 0, 120, 80, plate="96")
        self.assertEqual(centers.shape, (8, 12, 2))

    def test_well_centers_homography(self):
        quad = [(0, 0), (120, 0), (100, 80), (20, 80)]
        centers = PlateProcessor.well_centers(0, 0, 120, 80,
                                             plate="96",
                                             quad=quad)

        rows, cols = 8, 12
        src = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]],
                        np.float32)
        dst = np.array(quad, np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        xs = np.linspace(0.5, cols - 0.5, cols)
        ys = np.linspace(0.5, rows - 0.5, rows)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        homog = np.concatenate([grid, np.ones((grid.shape[0], 1))], axis=1)
        warped = (H @ homog.T).T
        expected = (warped[:, :2] / warped[:, 2:3]).reshape(rows, cols, 2)

        self.assertTrue(np.allclose(centers, expected))

    def test_srgb_roundtrip(self):
        rgb = np.array([[0, 128, 255]], dtype=np.float32)
        lin = srgb2lin(rgb)
        out = lin2srgb(lin)
        self.assertTrue(np.allclose(out, rgb, atol=1))

    def test_avg_rgb_basic(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:] = [0, 0, 255]
        centers = np.array([[[2, 2]]], dtype=float)
        rgb = PlateProcessor.avg_rgb(img, centers, win=3, trim=0)
        self.assertTrue(np.allclose(rgb[0][0], [255, 0, 0]))

    def test_gaussian_cluster_rgb_basic(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:] = [0, 255, 0]
        centers = np.array([[[2, 2]]], dtype=float)
        rgb = PlateProcessor.gaussian_cluster_rgb(img, centers, n=20, sigma=0.5)
        self.assertTrue(np.allclose(rgb[0][0], [0, 255, 0], atol=1))

    def test_baseline_saved_and_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = os.path.join(tmp, "snap.jpg")
            calib = os.path.join(tmp, "cal.json")
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            baseline = [[[1, 1, 1] for _ in range(12)] for _ in range(8)]
            baseline[0][0] = [2, 2, 2]
            baseline[0][1] = [0, 0, 0]
            raw_vals = [[[11, 11, 11] for _ in range(12)] for _ in range(8)]

            with open(snap, "wb") as f:
                f.write(b"0")

            with patch.object(PlateProcessor, "snapshot", return_value=snap), \
                 patch("camera.camera_w_calibration.cv2.imread", return_value=img), \
                 patch.object(PlateProcessor, "run_ui", return_value={
                     "rectangle": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                     "plate_type": "96",
                     "calibration_dots": [[0, 0]] * 24,
                     "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                     "baseline_colors": baseline,
                 }), \
                 patch.object(PlateProcessor, "gaussian_cluster_rgb", return_value=raw_vals), \
                 patch.object(PlateProcessor, "fit_rpcc", return_value=np.zeros((3, 10))), \
                 patch.object(PlateProcessor, "apply_rpcc", side_effect=lambda arr, M: np.array(arr)):
                corr = PlateProcessor().process_image(cam_index=0, snap=snap, calib=calib, force_ui=True)

                self.assertTrue(np.allclose(corr[0][0], [10, 10, 10]))
                with open(calib) as f:
                    saved = json.load(f)
                self.assertIn("baseline_colors", saved)


if __name__ == "__main__":
    unittest.main()

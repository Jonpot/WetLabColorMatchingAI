import importlib.util
import unittest

SKIP = importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None

if not SKIP:
    import numpy as np
    from camera.camera_w_calibration import PlateProcessor, srgb2lin, lin2srgb


@unittest.skipIf(SKIP, "numpy and cv2 are required")
class CameraUtilsTests(unittest.TestCase):
    def test_well_centers_shape(self):
        centers = PlateProcessor.well_centers(0, 0, 120, 80, plate="96")
        self.assertEqual(centers.shape, (8, 12, 2))

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


if __name__ == "__main__":
    unittest.main()

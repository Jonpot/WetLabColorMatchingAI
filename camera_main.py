# main.py  – quick demo
from camera_w_calibration import PlateProcessor
import numpy as np

processor = PlateProcessor()

# --- one‑liner: capture → (auto‑)calibrate → RGB cube -----------------
rgb_cube = processor.process_image(cam_index=1, warmup=15)    # e.g. (8,12,3)
print(rgb_cube[0][0], rgb_cube[0][1])

# # Optional statistics
# stats = processor.compute_rgb_statistics(rgb_matrix)
# if stats:
#     mean_rgb, std_rgb, max_d, min_d, avg_d = stats
#     print(mean_rgb, std_rgb)

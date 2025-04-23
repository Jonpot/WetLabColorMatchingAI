# main.py  – quick demo
from camera_w_calibration import PlateProcessor
import numpy as np

processor = PlateProcessor()

# Capture and process the image
raw_rgb_cube, corrected_rgb_cube = processor.process_image(cam_index=0, warmup=15)  # e.g., (8, 12, 3)

# Example: Access RGB values for a specific well (e.g., row 0, column 0)
row, col = 0, 2
raw_rgb = raw_rgb_cube[row][col]
corrected_rgb = corrected_rgb_cube[row][col]

print(f"Raw RGB values for well ({row}, {col}): {raw_rgb}")
print(f"Corrected RGB values for well ({row}, {col}): {corrected_rgb}")

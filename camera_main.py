from camera_w_calibration import PlateProcessor

processor = PlateProcessor()

# ‑‑> One‑liner: capture, (auto‑)calibrate, extract colors
rgb_matrix = processor.process_image(cam_index=0, warmup=5)
print(rgb_matrix)

# Optional statistics
stats = processor.compute_rgb_statistics(rgb_matrix)
if stats:
    mean_rgb, std_rgb, max_d, min_d, avg_d = stats
    print(mean_rgb, std_rgb)

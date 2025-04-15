from camera_w_calibration import PlateProcessor

processor = PlateProcessor()

rgb_matrix = processor.process_image(
    image_path="4.jpg",
    calib_filename="calibration.json",
    cam_index=0,
    warmup=5
)

print("RGB matrix shape:", rgb_matrix.shape)
# Optionally compute stats:
stats = processor.compute_rgb_statistics(rgb_matrix)
if stats is not None:
    mean_rgb, std_rgb, max_d, min_d, avg_d = stats
    print("Mean RGB:", mean_rgb)
    print("Std  RGB:", std_rgb)
    print("Max Dist:", max_d)
    print("Min Dist:", min_d)
    print("Avg Dist:", avg_d)



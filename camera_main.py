from camera_w_calibration import PlateProcessor
from ot2_utils import OT2Manager


robot = OT2Manager(hostname="169.254.122.0", username="root", key_filename="secret/ot2_ssh_key", password="lemos")

robot.add_turn_on_lights_action()
# robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
# robot.add_add_color_action(color_slot='8', plate_well="A2", volume=30)
robot.execute_actions_on_remote()
print("Post-execution 1")


processor = PlateProcessor()

# ‑‑> One‑liner: capture, (auto‑)calibrate, extract colors
rgb_matrix = processor.process_image(cam_index=1, warmup=5)
print(rgb_matrix)

# robot.add_close_action()
# robot.execute_actions_on_remote()






# # Optional statistics
# stats = processor.compute_rgb_statistics(rgb_matrix)
# if stats:
#     mean_rgb, std_rgb, max_d, min_d, avg_d = stats
#     print(mean_rgb, std_rgb)

from ot2_utils import OT2Manager
import numpy as np

try:
    # robot = OT2Manager(hostname="169.254.122.0", username="root", key_filename="secret/ot2_ssh_key", password="lemos", reduced_tips_info=3)
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None)
except Exception as e:
    print(f"Error initializing OT2Manager: {e}")
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None)

plate_col_letters = ["1", "2", "3", "4", "5", "6","7", "8", "9", "10", "11", "12"]
plate_rows_letters = ["A", "B", "C", "D", "E", "F", "G","H"]

# Split row letters into upper and lower halves
upper_rows = plate_rows_letters[:4]  # ['A', 'B', 'C', 'D']
lower_rows = plate_rows_letters[4:]  # ['E', 'F', 'G', 'H']

# Generate full well IDs
upper_half = [row + col for col in plate_col_letters for row in upper_rows]
upper_half_shuffle = np.random.shuffle(upper_half)
# lower_half = [row + col for col in plate_col_letters for row in lower_rows]

color_slots = ["7", "8", "9", "11"] # red green blue water
volumne_list = [200, 180, 120, 80, 20, 0]
volume_pair = [(v, 0) for v in volumne_list[:-1]]
color_pair = [(c1, c2) for c1 in color_slots for c2 in color_slots if c1 != c2]

# robot.add_turn_on_lights_action()
# Combine color pairs and volumes, allowing mirrored pairs
color_volume_pairs = []
for i, c1 in enumerate(color_slots):
    for j, c2 in enumerate(color_slots):
        if i != j:
            v1, v2 = volume_pair[i]
            color_volume_pairs.append((c1, c2, v1, v2))
color_volume_pairs = 4*color_volume_pairs

# Combine with shuffled upper_half wells
combined = list(zip(upper_half, color_volume_pairs))

# Final format: (well, color1, color2, vol1, vol2)
combined = [(well, c1, c2, v1, v2) for well, (c1, c2, v1, v2) in combined]
print(combined)

for i in combined:
    robot.add_add_color_action(color_slot=i[1], plate_well=i[0], volume=i[3])
    robot.add_add_color_action(color_slot=i[2], plate_well=i[0], volume=i[4])
    robot.execute_actions_on_remote()


robot.add_turn_off_lights_action()
robot.add_close_action()
robot.execute_actions_on_remote()


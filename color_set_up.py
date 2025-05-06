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

color_slots = ["7", "8", "9", "11"] # red green blue water
volume = [200, 180, 120, 80, 20]



# np.shuffle

robot.add_turn_on_lights_action()
robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
robot.add_add_color_action(color_slot='8', plate_well="A2", volume=30)
robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
robot.add_add_color_action(color_slot='8', plate_well="A2", volume=30)
robot.add_add_color_action(color_slot='9', plate_well="A3", volume=30)
robot.execute_actions_on_remote()

#robot.add_blink_lights_action(num_blinks=5)
robot.add_turn_off_lights_action()
# robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
robot.add_close_action()

robot.execute_actions_on_remote()
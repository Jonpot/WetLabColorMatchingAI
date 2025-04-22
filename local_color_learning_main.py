from ot2_utils import OT2Manager


robot = OT2Manager(hostname="169.254.122.0", username="root", key_filename="secret/ot2_ssh_key", password="lemos")

robot.add_turn_on_lights_action()
robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
robot.add_add_color_action(color_slot='8', plate_well="A2", volume=30)

robot.execute_actions_on_remote()
print("Post-execution 1")
robot.add_blink_lights_action(num_blinks=5)
robot.add_turn_off_lights_action()
robot.add_add_color_action(color_slot='7', plate_well="A1", volume=30)
robot.add_close_action()

robot.execute_actions_on_remote()
print("Post-execution 2")
from ot2_utils import OT2Manager


robot = OT2Manager(hostname="169.254.122.0", username="root", key_filename="secret/ot2_ssh_key", password="lemos")

robot.add_blink_lights_action(5)

robot.add_close_action()

robot.execute_actions_on_remote()
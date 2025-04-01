import sys
import json
import time
import paramiko
from scp import SCPClient
from typing import Any, Dict, Optional
import threading

class OT2Manager:
    def __init__(self, hostname: str, username: str, password: str, key_filename: str):        
        # OT2 robot connection details
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        
        # Set up the SSH client and load the private key
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.private_key = paramiko.RSAKey.from_private_key_file(self.key_filename, password=self.password)
        except Exception as e:
            print(f"Error loading private key from {self.key_filename}: {e}")
            sys.exit(1)
        
        try:
            print(f"Connecting to OT2 robot at {self.hostname}...")
            self.ssh.connect(self.hostname, username=self.username, pkey=self.private_key)
        except Exception as e:
            print(f"Error connecting to {self.hostname}: {e}")
            sys.exit(1)

        self.args = {"is_updated": False, "actions": []}
        self._save_args_to_file("args.json")
        self._upload_file("args.json")
        self._start_robot_listener()

    def _upload_file(self, local_path: str) -> None: 
        # Upload the JSON file using SCP
        try:
            print("Uploading file using SCP...")
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.put(local_path, remote_path=f"/root/{local_path}")
            print(f"Uploaded '{local_path}' to /root/ on the OT2 robot.")
        except Exception as e:
            print(f"Error during file upload using SCP: {e}")
            self.ssh.close()
            sys.exit(1)
        
    def _execute_command(self, command: str) -> None:
        # Execute the command on the OT2 robot to run the test_lights.py script
        print(f"Executing remote command: {command}")
        try:
            stdin, stdout, stderr = self.ssh.exec_command(command)
            output = stdout.read().decode()
            errors = stderr.read().decode()
            if output:
                print("Command output:")
                print(output)
            if errors:
                print("Command errors:")
                print(errors)
        except Exception as e:
            print(f"Error executing remote command: {e}")
        finally:
            self.ssh.close()
            print("SSH connection closed.")
        
    #def _start_robot_listener(self) -> None:
    #    # Start the robot listener in a separate thread
    #    def listener_thread():
    #        global finished_flag
    #        finished_flag = False
    #        command = "cd /root/ && opentrons_execute remote_ot2_color_learning_main.py"
    #        print(f"Executing remote command: {command}")
    #        try:
    #            stdin, stdout, stderr = self.ssh.exec_command(command)
    #            for line in iter(stdout.readline, ""):
    #                print(line.strip())
    #                if "Ready" in line.strip():
    #                    finished_flag = True
    #                    print("Finished flag set to True")
    #        except Exception as e:
    #            print(f"Error executing remote command: {e}")
    #        finally:
    #            self.ssh.close()
    #            print("SSH connection closed.")
#
    #    # Start the listener thread
    #    thread = threading.Thread(target=listener_thread, daemon=True)
    #    thread.start()

    def _start_robot_listener(self) -> None:
        # Start the robot listener in a separate thread
        global finished_flag
        finished_flag = False
        command = "cd /root/ && opentrons_execute remote_ot2_color_learning_main.py"
        print(f"Executing remote command: {command}")
        try:
            stdin, stdout, stderr = self.ssh.exec_command(command)
            for line in iter(stdout.readline, ""):
                print(line.strip())
                if "Ready" in line.strip():
                    finished_flag = True
                    print("Finished flag set to True")
        except Exception as e:
            print(f"Error executing remote command: {e}")
        finally:
            self.ssh.close()
            print("SSH connection closed.")

    def _add_action(self, action_name: str, action_value: Optional[Dict[str, Any]] = None) -> None:
        # Add an action to the args list
        if action_value is None:
            action_value = {}
        self.args["actions"].append({action_name: action_value})
        self.args["is_updated"] = True
        print(f"Added action: {action_name} with value: {action_value}")

    def _save_args_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.args, fp=f)
        print(f"Saved args to {filename}")

    def _listen_for_completion(self) -> None:
        # Wait for the robot to finish executing the commands
        while not finished_flag:
            print("Waiting for robot to finish...")
            time.sleep(5)

    def execute_actions_on_remote(self) -> None:
        global finished_flag
        # Save the args to a JSON file
        filename = "args.json"
        self._save_args_to_file(filename)

        # Upload the JSON file to the OT2 robot
        finished_flag = False
        self._upload_file(filename)

        # The server will automatically detect the new file and execute them
        # Block until the robot finishes executing the commands
        self._listen_for_completion()

    def add_blink_lights_action(self, num_blinks: int) -> None:
        # Add the blink lights action to the args list
        self._add_action("blink_lights", {"num_blinks": num_blinks})

    def add_close_action(self) -> None:
        # Add the close action to the args list
        self._add_action("close")

    def add_add_color_action(self, color_slot: str, plate_well: str, volume: float) -> None:
        # Add the add color action to the args list
        self._add_action("add_color", {"color_slot": color_slot, "plate_well": plate_well, "volume": volume})

    def __del__(self) -> None:
        # Ensure the SSH connection is closed when the object is deleted
        if self.ssh:
            self.ssh.close()
            print("SSH connection closed.")
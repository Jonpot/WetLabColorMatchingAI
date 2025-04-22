import sys
import json
import time
import paramiko
from scp import SCPClient
from typing import Any, Dict, Optional
import threading

class OT2Manager:
    def __init__(self, hostname: str, username: str, password: str, key_filename: str, virtual_mode: bool = False) -> None:
        self.virtual_mode = virtual_mode
        if not self.virtual_mode:
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

            # Initialize the args file and a flag to indicate when the remote process signals completion
            self.args = {"is_updated": False, "actions": []}
            self.finished_flag = False
            self.error_flag = False
            self._save_args_to_file("args.jsonx")
            self._upload_file("args.jsonx")
            self._start_robot_listener()
            self._listen_for_completion()
        print("OT2Manager initialized and ready.")
        input("Press Enter to continue and run the protocol...")


    def _upload_file(self, local_path: str) -> None:
        """Upload a file using SCP without closing the SSH connection."""
        try:
            print("Uploading file using SCP...")
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.put(local_path, remote_path=f"/root/{local_path}")
            print(f"Uploaded '{local_path}' to /root/ on the OT2 robot.")
        except Exception as e:
            print(f"Error during file upload using SCP: {e}")

    def _start_robot_listener(self) -> None:
        """
        Start a dedicated thread that opens an interactive shell,
        sends the environment setup and command to run the protocol,
        and continuously reads the output until a "Ready" signal is detected.
        """
        def listener():
            try:
                # Open an interactive shell session (PTY)
                channel = self.ssh.invoke_shell()
                print("Starting remote robot listener via interactive shell...")
                # Send commands to set the environment variable, change directory, and start the process.
                channel.send("export RUNNING_ON_PI=1\n")
                channel.send("cd /root/\n")

                # Check server status
                channel.send("STATUS=$(systemctl is-active opentrons-robot-server)\n")
                channel.send("echo Service is: $STATUS\n")
                channel.send("if [ \"$STATUS\" = \"active\" ]; then\n")
                channel.send("  echo Stopping robot server...\n")
                channel.send("  systemctl stop opentrons-robot-server\n")
                channel.send("else\n")
                channel.send("  echo Robot server is not running. No need to stop.\n")
                channel.send("fi\n")

                # Start the remote script
                channel.send("opentrons_execute remote_ot2_color_learning_main.py\n")
                
                # Continuously read output from the remote process.
                buffer = ""
                while True:
                    if channel.recv_ready():
                        output = channel.recv(1024).decode('utf-8')
                        buffer += output
                        # Process complete lines from the buffered output
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                print(line.strip())
                                if "Ready" in line:
                                    self.finished_flag = True
                                    print("Finished flag set to True")
                                elif "Error" in line:
                                    print("Error detected in remote process output.")
                                    self.finished_flag = True
                                    self.error_flag = True
                    # Exit loop if the channel is closed.
                    if channel.closed:
                        print("Remote shell channel closed.")
                        break
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in remote listener: {e}")

        listener_thread = threading.Thread(target=listener, daemon=True)
        listener_thread.start()

    def _add_action(self, action_name: str, action_value: Optional[Dict[str, Any]] = {}) -> None:
        """Add an action to the args list."""
        self.args["actions"].append({action_name: action_value})
        self.args["is_updated"] = True
        print(f"Added action: {action_name} with value: {action_value}")

    def _save_args_to_file(self, filename: str) -> None:
        """Save the current args to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.args, f)
        print(f"Saved args to {filename}")

    def _listen_for_completion(self) -> None:
        """Wait until the remote process signals 'Ready'.
        
        ::raises RuntimeError: If an error is detected during the remote process execution.
        """
        tries = 0
        while not self.finished_flag and not self.error_flag:
            tries += 1
            print(f"\rWaiting for robot to finish... [Attempt #{tries}]", end="")
            sys.stdout.flush()
            time.sleep(5)
        # Reset the flag for future operations
        print("Robot finished processing. Finished flag reset.")
        self.finished_flag = False

        if self.error_flag:
            print("Error detected during remote process execution.")
            self.error_flag = False
            raise RuntimeError("Error detected during remote process execution.")

    def execute_actions_on_remote(self) -> None:
        """
        Save and upload the args file and then wait for the remote process
        to signal that it is ready for new instructions.

        ::raises RuntimeError: If an error is detected during the remote process execution.
        """
        if self.virtual_mode:
            self.args["is_updated"] = False  # Reset the update flag for the next set of actions
            self.args["actions"] = []
            print("Running in virtual mode. Actions not executed on remote.")
            return  # Skip execution in virtual mode
        filename = "args.jsonx"
        self._save_args_to_file(filename)
        self._upload_file(filename)
        try:
            self._listen_for_completion()
        except RuntimeError as e:
            raise e

        self.args["is_updated"] = False  # Reset the update flag for the next set of actions
        self.args["actions"] = []
        print("Actions executed on remote. Ready for new instructions.")

    def add_blink_lights_action(self, num_blinks: int) -> None:
        """Queue a blink lights action."""
        self._add_action("blink_lights", {"num_blinks": num_blinks})

    def add_turn_on_lights_action(self) -> None:
        """Queue a turn on lights action."""
        self._add_action("turn_on_lights")

    def add_turn_off_lights_action(self) -> None:
        """Queue a turn off lights action."""
        self._add_action("turn_off_lights")

    def add_calibrate_96_well_plate(self) -> None:
        """Queue a calibrate 96 well plate action."""
        self._add_action("calibrate_96_well_plate")

    def add_close_action(self) -> None:
        """Queue a close action."""
        self._add_action("close")

    def add_add_color_action(self, color_slot: str, plate_well: str, volume: float) -> None:
        """Queue an add color action."""
        self._add_action("add_color", {"color_slot": color_slot, "plate_well": plate_well, "volume": volume})

    def __del__(self) -> None:
        # Close the SSH connection when the object is deleted.
        if self.ssh:
            self.ssh.close()
            print("SSH connection closed.")

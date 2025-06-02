import sys
import json
import time
import paramiko
from scp import SCPClient
from typing import Any, Dict, List, Optional
import threading


def get_plate_type(calibration_file: str = "camera/calibration.json") -> str:
    """Return the labware name from the calibration file.

    The camera calibration workflow writes the plate type to ``camera/calibration.json``
    as a numeric string (e.g. ``"96"``).  This helper converts that value into
    the corresponding Opentrons labware name.  If the file does not exist or is
    invalid, a 96-well plate is assumed.

    Parameters
    ----------
    calibration_file:
        Path to the calibration configuration produced by
        ``camera_w_calibration.py``.

    Returns
    -------
    str
        Labware identifier suitable for ``ProtocolContext.load_labware``.
    """

    mapping = {
        "12": "corning_12_wellplate_6.9ml_flat",
        "24": "corning_24_wellplate_3.4ml_flat",
        "48": "corning_48_wellplate_760ul_flat",
        "96": "corning_96_wellplate_360ul_flat",
    }

    try:
        with open(calibration_file, "r") as f:
            cfg = json.load(f)
            plate_key = str(cfg.get("plate_type", "96"))
    except (FileNotFoundError, json.JSONDecodeError):
        plate_key = "96"

    return mapping.get(plate_key, mapping["96"])

class WellFullError(Exception):
    """Exception raised when a well is full."""
    pass

class TiprackEmptyError(Exception):
    """Exception raised when the tip rack is empty."""
    pass


class OT2Manager:
    def __init__(self,
                 hostname: str,
                 username: str,
                 password: str,
                 key_filename: str,
                 plate_slot: str = '1',
                 ammo_slot: str = '2',
                 tiprack_slot: str = '3',
                 missile_volume: int = 100,
                 default_volume: int = 50,
                 virtual_mode: bool = False,
                 reduced_tips_info: None | int = 2,
                 bypass_startup_key: bool = False) -> None:
        self.virtual_mode = virtual_mode
        self.last_error_type = None
        self.reduced_tips_info = reduced_tips_info
        self.args = {"is_updated": False, "actions": [], "reduced_tips_info": self.reduced_tips_info, "plate_slot": plate_slot, "ammo_slot": ammo_slot, "tiprack_slot": tiprack_slot, "missile_volume": missile_volume, "default_volume": default_volume}
        self.finished_flag = False
        self.error_flag = False
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
            self._save_args_to_file("battleship/robot/args.jsonx")
            self._upload_file("battleship/robot/args.jsonx", "args.jsonx")
            # Ensure the latest protocol script is on the robot
            self._upload_file("battleship/remote/remote_ot2_battleship_main.py",
                              "remote_ot2_battleship_main.py")
            self._start_robot_listener()
            self._listen_for_completion()
        print("OT2Manager initialized and ready.")
        if not bypass_startup_key:
            input("Press Enter to continue and run the protocol...")


    def _upload_file(self, local_path: str, filename: str) -> None:
        """Upload a file using SCP without closing the SSH connection."""
        try:
            print("Uploading file using SCP...")
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.put(local_path, remote_path=f"/root/{filename}")
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
                channel.send("opentrons_execute remote_ot2_battleship_main.py\n")
                
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

                                    if "tip" in line:
                                        self.last_error_type = TiprackEmptyError
                                    elif "well" in line:
                                        self.last_error_type = WellFullError
                                    print(f"Last error type set to: {self.last_error_type}")
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
                               If this happens, it is useful to check the last_error_type attribute
                               to determine the type of error that occurred, likely either a
                               TiprackEmptyError or a WellFullError.
        """
        if self.virtual_mode:
            self.args["is_updated"] = False  # Reset the update flag for the next set of actions
            self.args["actions"] = []
            print("Running in virtual mode. Actions not executed on remote.")
            return  # Skip execution in virtual mode
        filename = "args.jsonx"
        filepath = f"battleship/robot/{filename}"
        self._save_args_to_file(filepath)
        self._upload_file(filepath, filename)
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

    def add_refresh_tiprack_action(self) -> None:
        """Queue a refresh tip rack action."""
        self._add_action("refresh_tiprack")

    def add_fire_missile_action(self, plate_well: str) -> None:
        """Queue a fire missile action."""
        self._add_action("fire_missile", {"plate_well": plate_well})

    def __del__(self) -> None:
        # Close the SSH connection when the object is deleted.
        try:
            self.ssh.close()
            print("SSH connection closed.")
        except Exception as e:
            print(f"Error closing SSH connection: {e}")

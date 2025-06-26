# OT2Manager API Documentation

## Overview

The `OT2Manager` class provides a high-level interface for remotely controlling an OT-2 robot via SSH. It leverages Paramiko and SCP for communication with the robot and employs an interactive shell session (with PTY) to mimic the behavior of a local terminal (such as PuTTY). The API enables you to upload an action file (`args.jsonx`), queue specific robot actions, and wait for the robot’s process to signal that it is ready for new instructions.

### Key Features

- **Persistent SSH Connection:** Maintains a stable and interactive SSH session.
- **Interactive Shell with PTY:** Mimics an interactive terminal environment for proper output and device access.
- **Threaded Output Listener:** Continuously monitors the remote process output for signals (e.g., "Ready") to synchronize execution.
- **Action Queue Mechanism:** Allows you to add and execute multiple actions remotely by updating a JSON file (`args.jsonx`).

---

## Requirements

- **Python 3.x**
- **Paramiko:** For SSH communication  
  *Installation:* `pip install paramiko`
- **scp:** For file transfers over SCP  
  *Installation:* `pip install scp`
- **A valid SSH key and credentials** to access the OT-2 robot.

---

## Class: OT2Manager

### Initialization

```
OT2Manager(hostname: str, username: str, password: str, key_filename: str)
```

**Parameters:**

- `hostname`: The IP address or hostname of the OT-2 robot.
- `username`: SSH username (typically "root").
- `password`: Password used to decrypt the private key file.
- `key_filename`: The path to the SSH private key file.

**What It Does:**
- Loads and validates the SSH private key.
- Establishes an SSH connection to the robot.
- Initializes an empty action list stored in a JSON file (`args.jsonx`).
- Uploads the initial `args.jsonx` file.
- Uploads the latest `remote_ot2_color_learning_main.py` protocol script.
- Starts an interactive shell session listener that sends the required initialization commands (sets `RUNNING_ON_PI=1` and runs the protocol script on the robot).

---

## Methods

### `_upload_file(local_path: str) -> None`

Uploads the specified file to the `/root/` directory on the OT-2 robot using SCP.  
**Usage:** Automatically called when the action file changes.

### `_start_robot_listener() -> None`

Starts a dedicated daemon thread that:
- Opens an interactive shell session.
- Sends the initialization commands:
  - `export RUNNING_ON_PI=1`
  - `cd /root/`
  - `opentrons_execute remote_ot2_color_learning_main.py`
- Continuously monitors output and sets a flag when “Ready” is detected.

**Usage:** Called during initialization to set up the robot process listener.

### `_listen_for_completion() -> None`

Blocks execution until the remote process signals it is ready for new actions. This method checks a shared flag (`finished_flag`) that the listener thread sets when the output contains “Ready.”

**Usage:** Used in `execute_actions_on_remote()` to wait for the robot to finish processing current commands.

### `execute_actions_on_remote() -> None`

Saves the current actions in the JSON file (`args.jsonx`), uploads the file to the robot, and waits for the robot’s process to signal that it is ready (using `_listen_for_completion()`).

**Usage Example:**

```
robot.execute_actions_on_remote()
```

### Action Queue Methods

These methods allow you to queue specific actions which are then written to the `args.jsonx` file before you trigger a remote execution.

- **`add_turn_on_lights_action() -> None`**  
  Queues an action to turn on the robot’s lights.

- **`add_turn_off_lights_action() -> None`**  
  Queues an action to turn off the robot’s lights.

- **`add_blink_lights_action(num_blinks: int) -> None`**  
  Queues an action to blink the lights a specified number of times.

  **Parameters:**  
  - `num_blinks`: Number of times to blink.

- **`add_add_color_action(color_well: str, plate_well: str, volume: float) -> None`**  
  Queues an action to add color with specified parameters.

  **Parameters:**  
  - `color_well`: The slot identifier for the color.
  - `plate_well`: The plate well location (e.g., "A1").
  - `volume`: The volume to add (in microliters or the applicable unit).

- **`add_close_action() -> None`**  
  Queues the action to close the session or terminate the protocol on the OT-2 robot.

### Destructor

- **`__del__() -> None`**

Ensures that the SSH connection is properly closed when the `OT2Manager` instance is destroyed.

---

## Usage Example

Below is an example showing how your team can use the API to control the OT-2 robot:

```
from ot2_utils import OT2Manager

# Initialize the OT2Manager with connection details
robot = OT2Manager(
    hostname="169.254.122.0",
    username="root",
    key_filename="secret/ot2_ssh_key",
    password="lemos"
)

# Queue initial actions (turn on lights and add colors)
robot.add_turn_on_lights_action()
robot.add_add_color_action(color_well='7', plate_well="A1", volume=30)
robot.add_add_color_action(color_well='8', plate_well="A2", volume=30)

# Execute the queued actions remotely and wait for completion
robot.execute_actions_on_remote()

# Queue follow-up actions (blink lights, turn off lights, add more color, close protocol)
robot.add_blink_lights_action(num_blinks=5)
robot.add_turn_off_lights_action()
robot.add_add_color_action(color_well='7', plate_well="A1", volume=30)
robot.add_close_action()

# Execute the new set of actions
robot.execute_actions_on_remote()
```

### Workflow Summary

1. **Initialization:**  
   The instance connects to the OT-2 robot and starts an interactive shell that automatically sets up the required environment.

2. **Action Queueing:**  
   The team can queue multiple actions using the provided methods.

3. **File Upload & Execution:**  
   Calling `execute_actions_on_remote()`:
   - Saves the current action list to `args.jsonx`.
   - Uploads the file to the robot.
   - Waits for a “Ready” signal from the robot before proceeding.
   
4. **Output Monitoring:**  
   The dedicated listener thread continuously displays output from the remote process to help with debugging and confirmation.

---

## Error Handling & Debugging

- **SSH Connection Errors:**  
  Initialization will exit if the SSH connection or key loading fails.
  
- **SCP Upload Failures:**  
  Errors during file upload are logged. Ensure network connectivity and permissions are set correctly.

- **Remote Listener Issues:**  
  The listener thread logs any exceptions. Validate that the remote script (`opentrons_execute remote_ot2_color_learning_main.py`) is accessible and executable.

- **Timeouts and Hangs:**  
  If the robot never outputs “Ready,” the `_listen_for_completion()` method will continue polling. Consider adding a maximum timeout for production use.

---

## Contribution Guidelines

- **Modifying Behavior:**  
  If adding new actions or improving the communication protocol, ensure that any modifications maintain the interactive shell behavior and proper synchronization.

- **Testing:**  
  Always test modifications on a staging OT-2 setup to confirm that remote commands are executed as expected.

- **Documentation:**  
  Update this documentation with any changes to the API. Include usage examples and note any caveats with the hardware interface.

---

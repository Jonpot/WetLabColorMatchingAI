#!/usr/bin/env python3
import sys
import json
import paramiko
import getpass
from scp import SCPClient

def main():
    # Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: blink_ot2_lights.py <num_blinks>")
        sys.exit(1)
    
    try:
        num_blinks = int(sys.argv[1])
    except ValueError:
        print("Error: <num_blinks> must be an integer.")
        sys.exit(1)
    
    # Create the JSON file "args.json" with the specified structure
    args_data = {"num_blinks": num_blinks}
    json_filename = "args.json"
    with open(json_filename, "w") as f:
        json.dump(args_data, f)
    print(f"Created JSON file '{json_filename}' with contents: {args_data}")
    
    # OT2 robot connection details
    hostname = "169.254.122.0"
    username = "root"
    key_filename = "ot2_ssh_key"  # using the OpenSSH key file
    
    # Set up the SSH client and load the private key
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        private_key = paramiko.RSAKey.from_private_key_file(key_filename, password='lemos')
    except Exception as e:
        print(f"Error loading private key from {key_filename}: {e}")
        sys.exit(1)
    
    try:
        print(f"Connecting to OT2 robot at {hostname}...")
        ssh.connect(hostname, username=username, pkey=private_key)
    except Exception as e:
        print(f"Error connecting to {hostname}: {e}")
        sys.exit(1)
    
    # Upload the JSON file using SCP (instead of SFTP)
    try:
        print("Uploading file using SCP...")
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(json_filename, remote_path=f"/root/{json_filename}")
        print(f"Uploaded '{json_filename}' to /root/ on the OT2 robot.")
    except Exception as e:
        print(f"Error during file upload using SCP: {e}")
        ssh.close()
        sys.exit(1)
    
    # Execute the command on the OT2 robot to run the test_lights.py script
    remote_command = "cd /root/ && opentrons_execute test_lights.py"
    print(f"Executing remote command: {remote_command}")
    try:
        stdin, stdout, stderr = ssh.exec_command(remote_command)
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
        ssh.close()
        print("SSH connection closed.")

if __name__ == '__main__':
    main()

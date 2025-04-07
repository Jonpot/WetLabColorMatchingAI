import cv2
from ot2_utils import OT2Manager
from camera_process import CameraImageProcessor

def main():
    # # Connect to the OT-2 robot
    # print("Connecting to OT2 robot...")
    # robot = OT2Manager(
    #     hostname="172.26.192.201",
    #     username="root",
    #     key_filename="C:/Users/shich/.ssh/ot2_ssh_key",
    #     password=""
    # )
    
    # # Add a blink lights action to the robot and execute it remotely
    # robot.add_blink_lights_action(num_blinks=5)
    # robot.execute_actions_on_remote()
    
    # Create an instance of the camera processor to capture and process an image
    camera = CameraImageProcessor()
    
    # Define adjustable camera properties (e.g. resolution)
    adjustable_properties = {
        cv2.CAP_PROP_FRAME_WIDTH: 1920,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080,
    }
    
    # Capture a snapshot using the specified camera index
    snapshot_path = camera.take_snapshot(
        cam_index=1,
        save_path="snapshot.jpg",
        warmup_frames=10,
        properties=adjustable_properties
    )
    
    # Process the captured image to obtain segmentation, well centers, and the RGB matrix
    segmented, centers, rgb_matrix, original = camera.process_image(
        image_path=snapshot_path,
        debug=True,
        plate_type="96"
    )
    
    if rgb_matrix is not None:
        print("RGB matrix shape:", rgb_matrix.shape)
        print(rgb_matrix)
        rgb_stats = camera.compute_rgb_statistics(rgb_matrix)
        print("RGB Statistics:", rgb_stats)
    
    # # Add a close action (if your workflow requires closing the session on the OT-2)
    # robot.add_close_action()
    # robot.execute_actions_on_remote()

if __name__ == "__main__":
    main()

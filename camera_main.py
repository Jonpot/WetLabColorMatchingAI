import cv2
from ot2_utils import OT2Manager
from camera_process import CameraImageProcessor

def main():
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

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import cv2
import time
import numpy as np
from robotpy_apriltag import AprilTagDetector
from scipy.spatial.distance import pdist, squareform

class CameraImageProcessor:
    def __init__(self):
        # Initialize the processor if needed (e.g., default settings)
        pass

    # --------- Camera Snapshot Methods ---------

    @staticmethod
    def list_cameras(max_tested=10):
        """
        Check camera indices 0 to max_tested-1 using the DirectShow backend.
        Returns a list of valid camera indices.
        """
        valid_indices = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    valid_indices.append(idx)
                cap.release()
        return valid_indices

    @staticmethod
    def set_camera_properties(cap, properties):
        """
        Set camera properties based on the provided dictionary.
        'properties' is a dictionary where keys are cv2 property constants
        and values are the desired settings.
        """
        for prop, value in properties.items():
            cap.set(prop, value)
            current_val = cap.get(prop)
            print(f"Set property {prop} to {value}. Current value: {current_val}")

    @staticmethod
    def take_snapshot(cam_index=1, save_path="snapshot.jpg", warmup_frames=10, properties=None):
        """
        Open the specified camera (using DirectShow), enable auto white balance,
        apply adjustable properties, discard warmup frames, and capture & save a snapshot.
        Returns the path where the snapshot is saved.
        """
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        # Enable auto white balance
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        print("Auto white balance enabled:", cap.get(cv2.CAP_PROP_AUTO_WB))
        # Set additional camera properties if provided
        if properties:
            CameraImageProcessor.set_camera_properties(cap, properties)
        # Warm up the camera by discarding a few frames
        for i in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                print("Warning: Warm-up frame not captured.")
            time.sleep(0.1)
        # Capture the final frame as snapshot
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"Snapshot taken and saved as {save_path}")
        else:
            print("Failed to capture a snapshot.")
        cap.release()
        return save_path

    # --------- Image Processing Methods ---------

    @staticmethod
    def resize_to_fit(img, max_width=1280, max_height=720):
        """
        Resize image while maintaining aspect ratio so that it does not exceed the given dimensions.
        Returns the resized image and the scaling factor.
        """
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return resized_img, scale

    @staticmethod
    def segment_by_apriltags(path, debug=False):
        """
        Read the image from the given path, detect AprilTags to determine the region of interest,
        apply a cropping buffer, and return the segmented image along with the top-left and
        bottom-right coordinates, and the original image.
        """
        original_bgr = cv2.imread(path)
        if original_bgr is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        detector = AprilTagDetector()
        detector.addFamily("tag36h11")
        results = detector.detect(gray)
        if len(results) < 2:
            print(f"Not enough AprilTags detected. Found {len(results)}")
            return None, (None, None), (None, None), original_bgr

        def get_center(det):
            pt = det.getCenter()
            return (pt.x, pt.y)

        top_left_det = min(results, key=lambda r: sum(get_center(r)))
        bottom_right_det = max(results, key=lambda r: sum(get_center(r)))
        top_left = get_center(top_left_det)
        bottom_right = get_center(bottom_right_det)
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        # Apply cropping buffer (adjust these values as needed)
        x1 += 260
        y1 += 145
        x2 -= 230
        y2 -= 200

        if debug:
            debug_img = original_bgr.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            resized, _ = CameraImageProcessor.resize_to_fit(debug_img)
            cv2.imshow("AprilTag Rectangle with Buffer", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        segmented = original_bgr[y1:y2, x1:x2]
        return segmented, (x1, y1), (x2, y2), original_bgr

    @staticmethod
    def get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type="96"):
        """
        Compute the centers of wells in a grid within the specified bounding box.
        Plate types can be "12", "24", "48", or "96" (default is "96").
        Returns a list of (x, y) center coordinates.
        """
        layouts = {"12": (8, 12), "24": (4, 6), "48": (6, 8), "96": (8, 12)}
        rows, cols = layouts.get(str(plate_type), layouts["96"])
        width = x2 - x1
        height = y2 - y1
        step_x = width / cols
        step_y = height / rows
        centers = []
        for i in range(rows):
            for j in range(cols):
                cx = int(x1 + (j + 0.5) * step_x)
                cy = int(y1 + (i + 0.5) * step_y)
                centers.append((cx, cy))
        return centers

    @staticmethod
    def extract_rgb_values(image, centers, x_offset=0, y_offset=0):
        """
        For each center coordinate, sample the RGB values at the center and its 4-connected neighbors.
        Returns an array (3 x N) of averaged RGB values (converted from BGR to RGB).
        """
        h, w = image.shape[:2]
        rgb_values = []
        for cx, cy in centers:
            local_cx = cx - x_offset
            local_cy = cy - y_offset
            pixels = []
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                x, y = local_cx + dx, local_cy + dy
                if 0 <= x < w and 0 <= y < h:
                    bgr = image[y, x]
                    pixels.append(bgr)
            if pixels:
                avg = np.mean(pixels, axis=0)
                rgb = [avg[2], avg[1], avg[0]]  # Convert BGR to RGB
                rgb_values.append(rgb)
            else:
                rgb_values.append([0, 0, 0])
        rgb_matrix = np.array(rgb_values).T  # shape: (3, N)
        return rgb_matrix

    def process_image(self, image_path, debug=False, plate_type="96"):
        """
        Process the image at the specified path:
          1. Segment the image based on AprilTag detection.
          2. Compute well center coordinates (e.g., for a 96-well plate).
          3. Extract the RGB color matrix from the segmented image.
        Returns:
          segmented: segmented image
          centers: list of well center coordinates
          rgb_matrix: color matrix (3 x N)
          original: original image
        """
        segmented, (x1, y1), (x2, y2), original = CameraImageProcessor.segment_by_apriltags(image_path, debug=debug)
        if segmented is None:
            print("AprilTag segmentation failed.")
            return None, None, None, original

        centers = CameraImageProcessor.get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type=plate_type)
        rgb_matrix = CameraImageProcessor.extract_rgb_values(segmented, centers, x_offset=x1, y_offset=y1)
        return segmented, centers, rgb_matrix, original

    @staticmethod
    def compute_rgb_statistics(selected_rgbs):
        """
        Compute statistical measures on the selected RGB values:
          - Mean RGB
          - Standard deviation of RGB
          - Maximum, minimum, and average pairwise Euclidean distances
        Returns a tuple: (mean_rgb, std_rgb, max_distance, min_distance, average_distance)
        """
        mean_rgb = np.mean(selected_rgbs, axis=0)
        std_rgb = np.std(selected_rgbs, axis=0)
        dist_matrix = squareform(pdist(selected_rgbs))
        max_distance = np.max(dist_matrix)
        min_distance = np.min(dist_matrix)
        average_distance = np.mean(dist_matrix[np.triu_indices(len(selected_rgbs), k=1)])
        return mean_rgb, std_rgb, max_distance, min_distance, average_distance

# # Testing code (for module standalone testing)
# if __name__ == "__main__":
#     processor = CameraImageProcessor()
    
#     # 1. Capture a snapshot using the camera with adjustable properties (e.g., resolution)
#     adjustable_properties = {
#         cv2.CAP_PROP_FRAME_WIDTH: 1920,
#         cv2.CAP_PROP_FRAME_HEIGHT: 1080,
#     }
#     snapshot_path = processor.take_snapshot(cam_index=1, save_path="snapshot.jpg", warmup_frames=10, properties=adjustable_properties)
    
#     # 2. Process the captured image to obtain segmentation, well centers, and RGB matrix
#     segmented, centers, rgb_matrix, original = processor.process_image(snapshot_path, debug=True, plate_type="96")
#     rgb_static = processor.compute_rgb_statistics(rgb_matrix)
#     if rgb_matrix is not None:
#         print("RGB matrix shape:", rgb_matrix.shape)
#         print(rgb_matrix)
#         # print(rgb_static)

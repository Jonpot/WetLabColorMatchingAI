import cv2
import time
import numpy as np
import json
import os

class PlateProcessor:
    def __init__(self):
        # UI / calibration properties
        self.points = []
        self.dragging_idx = -1
        self.img_copy = None
        self.confirmed = False
        self.CONFIRM_BTN_TOPLEFT = None
        self.BTN_WIDTH = 140
        self.BTN_HEIGHT = 30

    # ----------------------------------------------------------------
    # ----------------------- Camera Methods --------------------------
    # ----------------------------------------------------------------

    @staticmethod
    def list_cameras(max_tested=10):
        """
        Check camera indices [0..max_tested-1] using the DirectShow backend.
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
        Set camera properties (e.g., resolution) via a dictionary:
        { cv2.CAP_PROP_FRAME_WIDTH: 1920,
          cv2.CAP_PROP_FRAME_HEIGHT: 1080, ... }
        """
        for prop, value in properties.items():
            cap.set(prop, value)
            current_val = cap.get(prop)
            print(f"Set property {prop} to {value}. Current value: {current_val}")

    @staticmethod
    def take_snapshot(cam_index=0, save_path="snapshot.jpg", warmup_frames=10, properties=None):
        """
        Open the specified camera (DirectShow on Windows), enable auto white balance,
        apply camera properties, discard warmup frames, capture & save a snapshot.

        Returns the path where the snapshot is saved.
        """
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        # Enable auto white balance
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        print("Auto white balance enabled:", cap.get(cv2.CAP_PROP_AUTO_WB))

        if properties:
            PlateProcessor.set_camera_properties(cap, properties)

        # Discard several frames for warming up
        for i in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                print("Warning: Warm-up frame not captured.")
            time.sleep(0.1)

        # Capture the final frame
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"Snapshot taken and saved as '{save_path}'")
        else:
            print("Failed to capture a snapshot.")
        cap.release()
        return save_path

    # ----------------------------------------------------------------
    # -------------------- Utility / Image Methods --------------------
    # ----------------------------------------------------------------

    @staticmethod
    def resize_to_fit(img, max_width=1280, max_height=720):
        """
        Resize 'img' so it does not exceed max_width or max_height,
        preserving aspect ratio. Returns (resized_img, scale_factor).
        """
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return resized_img, scale

    @staticmethod
    def plate_from_trackbar(val):
        """
        Map trackbar value (0..3) to a string plate type: 0->"12", 1->"24", 2->"48", 3->"96".
        """
        mapping = {0: "12", 1: "24", 2: "48", 3: "96"}
        return mapping.get(val, "96")

    @staticmethod
    def get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type="96"):
        """
        Compute the (x, y) center coordinates for each well in the specified bounding box.
        Plate types can be "12", "24", "48", or "96".
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
    def extract_rgb_values(image, centers, rows=8, cols=12, x_offset=0, y_offset=0):
        """
        For each center (cx, cy), sample the color in a small 5‑pixel region
        (the center + its 4 neighbors). Compute the average BGR->RGB and
        return a Python list-of-lists of shape (rows x cols), where each
        entry is [R, G, B] for that well.
        
        Arguments:
        - image:       OpenCV image array (H x W x 3, BGR)
        - centers:     flat list of (cx, cy) tuples, length should be rows*cols
        - rows, cols:  desired output matrix dimensions
        - x_offset, y_offset: if your centers are in a sub-image, use offsets
        
        Returns:
        - rgb_matrix:  list of length `rows`, each an inner list of length `cols`;
                       rgb_matrix[r][c] == [avg_R, avg_G, avg_B] for that well
        """
        h, w = image.shape[:2]
        if len(centers) != rows * cols:
            raise ValueError(f"Expected {rows*cols} centers, got {len(centers)}")
        
        flat_rgb = []
        for cx, cy in centers:
            local_cx = cx - x_offset
            local_cy = cy - y_offset

            samples = []
            for dx, dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                x = local_cx + dx
                y = local_cy + dy
                if 0 <= x < w and 0 <= y < h:
                    b, g, r = image[y, x]
                    samples.append((r, g, b))
            if samples:
                avg_r, avg_g, avg_b = np.mean(samples, axis=0)
            else:
                avg_r = avg_g = avg_b = 0.0
            b,g,r = image[local_cx, local_cy]
            flat_rgb.append([float(r), float(g), float(b)])

        # reshape flat list into rows x cols
        rgb_matrix = []
        for r in range(rows):
            row_vals = flat_rgb[r*cols : (r+1)*cols]
            rgb_matrix.append(row_vals)

        return rgb_matrix
    
    @staticmethod
    def compute_rgb_statistics(rgb_matrix_transposed):
        """
        Given a (3 x N) matrix, first transpose it to (N x 3),
        then compute mean, std, and pairwise distances (max, min, avg).
        Requires scipy for pdist, squareform.
        """
        if rgb_matrix_transposed is None or rgb_matrix_transposed.size == 0:
            return None

        selected_rgbs = rgb_matrix_transposed.T  # (N, 3)
        mean_rgb = np.mean(selected_rgbs, axis=0)
        std_rgb = np.std(selected_rgbs, axis=0)

        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(selected_rgbs))
        max_distance = np.max(dist_matrix)
        min_distance = np.min(dist_matrix)
        # average distance (upper triangle only, to avoid duplicates)
        avg_distance = np.mean(dist_matrix[np.triu_indices(len(selected_rgbs), k=1)])
        return (mean_rgb, std_rgb, max_distance, min_distance, avg_distance)

    # ----------------------------------------------------------------
    # --------------------- Calibration UI ----------------------------
    # ----------------------------------------------------------------

    def draw_ui(self, disp):
        """
        Draw corner points, bounding lines, well centers, a Confirm button,
        and text instructions on the display image.
        """
        h, w = disp.shape[:2]

        # Draw instructions at the top-left
        instructions = [
            "Drag corners to define plate region",
            "Use trackbar: (0=12,1=24,2=48,3=96)",
            "Click 'Confirm' or press 'c' to finalize",
            "Press ESC to cancel"
        ]
        y_offset = 25
        for line in instructions:
            cv2.putText(disp, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30

        # Draw corner points and numbering
        for i, pt in enumerate(self.points):
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
            cv2.putText(disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # If 4 points, draw lines and show sample well centers
        if len(self.points) == 4:
            for i in range(4):
                cv2.line(disp, self.points[i],
                         self.points[(i + 1) % 4], (0, 255, 0), 2)
            xs = [pt[0] for pt in self.points]
            ys = [pt[1] for pt in self.points]
            rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)

            # Draw bounding rectangle
            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

            # Draw well centers
            plate_val = cv2.getTrackbarPos("Plate", "Calibration")
            plate_type = self.plate_from_trackbar(plate_val)
            centers = self.get_well_centers_boxed_grid(rx1, ry1, rx2, ry2, plate_type)
            for (cx, cy) in centers:
                cv2.circle(disp, (cx, cy), 3, (0, 0, 255), -1)

        # Draw confirm button in bottom-right
        self.CONFIRM_BTN_TOPLEFT = (w - self.BTN_WIDTH - 10, h - self.BTN_HEIGHT - 10)
        bx, by = self.CONFIRM_BTN_TOPLEFT
        cv2.rectangle(disp, (bx, by), (bx + self.BTN_WIDTH, by + self.BTN_HEIGHT),
                      (50, 205, 50), -1)
        cv2.putText(disp, "Confirm", (bx + 10, by + self.BTN_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return disp

    def update_windows(self):
        """
        Re-draws the calibration display window with corner points,
        bounding lines, and the confirm button.
        """
        disp = self.img_copy.copy()
        disp = self.draw_ui(disp)
        cv2.imshow("Calibration", disp)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse event callback for dragging corners or clicking the 'Confirm' button.
        Also clamps dragged corners to remain within image boundaries.
        """
        h, w = self.img_copy.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the Confirm button is clicked
            if self.CONFIRM_BTN_TOPLEFT:
                bx, by = self.CONFIRM_BTN_TOPLEFT
                if (bx <= x <= bx + self.BTN_WIDTH and
                        by <= y <= by + self.BTN_HEIGHT):
                    self.confirmed = True
                    return

            # Check if near an existing corner
            for i, pt in enumerate(self.points):
                # If mouse is within 10px of a corner, pick it
                if np.hypot(x - pt[0], y - pt[1]) < 10:
                    self.dragging_idx = i
                    return

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx != -1:
            # Clamp corners inside the image
            clamped_x = max(0, min(x, w - 1))
            clamped_y = max(0, min(y, h - 1))
            self.points[self.dragging_idx] = (clamped_x, clamped_y)
            self.update_windows()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

    def run_calibration_ui(self, resized_img):
        """
        Launches the calibration window. The user can:
        - Drag each of the 4 corners
        - Use the trackbar for plate type
        - Click 'Confirm' or press 'c' to finalize
        - Press ESC to cancel
        Returns a dict with {'rectangle': {...}, 'plate_type': ...},
        or None if canceled or timed out (3 mins).
        """
        self.img_copy = resized_img.copy()
        h, w = resized_img.shape[:2]

        # Set default corners (10% inset)
        margin_w, margin_h = int(w * 0.1), int(h * 0.1)
        self.points = [
            (margin_w, margin_h),
            (w - margin_w, margin_h),
            (w - margin_w, h - margin_h),
            (margin_w, h - margin_h),
        ]
        self.confirmed = False
        self.dragging_idx = -1

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        cv2.createTrackbar("Plate", "Calibration", 3, 3, lambda v: self.update_windows())
        self.update_windows()

        print("=== Calibration UI ===")
        print(" - Drag corners to match your plate edges.")
        print(" - Trackbar sets plate type (12,24,48,96).")
        print(" - Click 'Confirm' or press 'c' to finalize.")
        print(" - ESC or 3-minute timeout to cancel.\n")

        start_time = time.time()
        TIMEOUT_SECONDS = 180  # 3 minutes

        while True:
            key = cv2.waitKey(20) & 0xFF

            # If 3 minutes pass without confirmation, cancel calibration
            if time.time() - start_time > TIMEOUT_SECONDS:
                print("Calibration timed out. No confirmation within 3 minutes.")
                self.confirmed = False
                break

            if key == 27:  # ESC
                print("Calibration canceled by user (ESC).")
                self.confirmed = False
                break
            elif key == ord('c'):
                # Press 'c' to confirm
                self.confirmed = True
                break

            if self.confirmed:
                break

        xs = [pt[0] for pt in self.points]
        ys = [pt[1] for pt in self.points]
        rx1, ry1, rx2, ry2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        plate_val = cv2.getTrackbarPos("Plate", "Calibration")
        plate_type = self.plate_from_trackbar(plate_val)

        cv2.destroyWindow("Calibration")

        if not self.confirmed:
            # Return None if user canceled or timed out
            return None

        return {
            "rectangle": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "plate_type": plate_type
        }


    # ----------------------------------------------------------------
    # ---------------------- Calibration Methods ----------------------
    # ----------------------------------------------------------------

    def calibrate_from_file(self, image_path, calib_filename="calibration.json"):
        """
        Calibrate using an existing image file.
        If a calibration file exists, the user is prompted whether to reuse it.
        If not, run the calibration UI and save results.
        Returns (calib_data, resized_img, scale).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        resized_img, scale = self.resize_to_fit(img, 1280, 720)

        # Check for existing JSON
        if os.path.exists(calib_filename):
            use_saved = input("Calibration data exists. Use saved calibration? (y/n): ").strip().lower()
            if use_saved != "y":
                os.remove(calib_filename)
                print("Old calibration data deleted. Recalibrating...")
                calib_data = self.run_calibration_ui(resized_img)
                if calib_data is None:
                    print("No calibration data saved (user canceled).")
                    return None, resized_img, scale
                with open(calib_filename, "w") as f:
                    json.dump(calib_data, f, indent=4)
            else:
                with open(calib_filename, "r") as f:
                    calib_data = json.load(f)
        else:
            # No existing file, run calibration UI
            print("No calibration file found; running calibration UI...")
            calib_data = self.run_calibration_ui(resized_img)
            if calib_data is None:
                print("No calibration data saved (user canceled).")
                return None, resized_img, scale

            with open(calib_filename, "w") as f:
                json.dump(calib_data, f, indent=4)
            print("Calibration data saved.")

        return calib_data, resized_img, scale

    def calibrate_from_camera(self, cam_index=0, snapshot_path="calibration_pic.jpg",
                              calib_filename="calibration.json", warmup=10):
        """
        1) Take a snapshot from the specified camera index.
        2) Then run calibrate_from_file() on that snapshot.
        Returns (calib_data, resized_img, scale).
        """
        print(f"Taking snapshot from camera index {cam_index}...")
        pic_path = self.take_snapshot(cam_index, snapshot_path, warmup_frames=warmup)
        return self.calibrate_from_file(pic_path, calib_filename)

    # ----------------------------------------------------------------
    # ------------------- Modified process_image ----------------------
    # ----------------------------------------------------------------

    def process_image(
            self,
            cam_index: int = 0,
            warmup: int = 10,
            image_path: str | None = None,
            calib_filename: str = "calibration.json",
            snapshot_file: str = "snapshot.jpg" 
    ) -> np.ndarray:
        """
        • Always capture a fresh frame into SNAPSHOT_FILE (constant name).
        • If calibration data is missing, launch the UI once, then reuse it.
        • Return a 3 × N RGB matrix for the current plate.
        """

        # ------------------------------------------------------------------
        # 1)  Capture — always overwrite the constant file
        # ------------------------------------------------------------------
        if image_path is None:
            image_path = snapshot_file         # <── use the constant

        self.take_snapshot(
            cam_index=cam_index,
            save_path=image_path,
            warmup_frames=warmup,
        )

        # ------------------------------------------------------------------
        # 2)  Ensure calibration data exists
        # ------------------------------------------------------------------
        if not os.path.exists(calib_filename):
            print("[INFO] No calibration file found – entering calibration UI…")
            calib_data, _, _ = self.calibrate_from_file(
                image_path=image_path,
                calib_filename=calib_filename,
            )
            if calib_data is None:
                raise RuntimeError("Calibration aborted – no data saved.")
        else:
            with open(calib_filename, "r") as f:
                calib_data = json.load(f)

        # ------------------------------------------------------------------
        # 3)  Extract the RGB matrix
        # ------------------------------------------------------------------
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read snapshot: {image_path}")

        resized_img, _ = self.resize_to_fit(img, 1280, 720)

        rect = calib_data["rectangle"]
        rx1, ry1, rx2, ry2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
        plate_type = calib_data["plate_type"]

        centers = self.get_well_centers_boxed_grid(
            rx1, ry1, rx2, ry2, plate_type
        )
        rgb_matrix = self.extract_rgb_values(
            resized_img, centers, x_offset=rx1, y_offset=ry1
        )
        return rgb_matrix




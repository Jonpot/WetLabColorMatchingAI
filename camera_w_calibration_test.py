"""
Plate-imaging pipeline WITH learning-based WB and **no resizing**.
Requires OpenCV-contrib (`pip install opencv-contrib-python`).
"""
import cv2, time, json, os
import numpy as np
from pathlib import Path

class PlateProcessor:
    # ───────────────────────────────────────────── initialisation ─────────────────────────────────────────────
    def __init__(self):
        self.points, self.calibration_dots = [], [(50, 50), (100, 50), (150, 50), (200, 50)]
        self.dragging_idx = self.dragging_calib_idx = -1
        self.img_copy, self.confirmed = None, False
        self.CONFIRM_BTN_TOPLEFT, self.BTN_WIDTH, self.BTN_HEIGHT = None, 140, 30

    # ───────────────────────────────────────── advanced white balance ─────────────────────────────────────────
    @staticmethod
    def advanced_white_balance(img: np.ndarray) -> np.ndarray:
        """OpenCV learning-based automatic WB."""
        wb = cv2.xphoto.createLearningBasedWB()
        wb.setSaturationThreshold(0.98)          # ignore saturated highlights
        return wb.balanceWhite(img)

    # ───────────────────────────────────────── camera capture ─────────────────────────────────────────
    def take_snapshot(
        self, cam_index: int = 0, save_path: str = "snapshot.jpg",
        *, warmup_frames: int = 15, burst: int = 5,
        resolution: tuple[int, int] | None = (1600, 1200),
        auto_exposure: bool = True, exposure: float | None = None,
        gain: float | None = None, apply_advanced_wb: bool = True,
        properties: dict[int, float] | None = None
    ) -> str:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if resolution:
            w, h = resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.2)

        for _ in range(warmup_frames): cap.read(); time.sleep(0.05)

        acc = None
        for _ in range(max(burst, 1)):
            ret, frm = cap.read()
            if not ret: raise RuntimeError("Camera read failed")
            acc = frm.astype(np.float32) if acc is None else acc + frm
            time.sleep(0.03)
        cap.release()

        frame = (acc / burst).astype(np.uint8)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[Snapshot] {frame.shape[1]}×{frame.shape[0]} saved → {save_path}")
        return save_path

    # ───────────────────────────────────────── utility helpers (unchanged) ─────────────────────────────────────────
    @staticmethod
    def plate_from_trackbar(val): return {0: "12", 1: "24", 2: "48", 3: "96"}.get(val, "96")

    @staticmethod
    def get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type="96"):
        layouts = {"12": (8, 12), "24": (4, 6), "48": (6, 8), "96": (8, 12)}
        rows, cols = layouts.get(str(plate_type), layouts["96"])
        dx, dy = (x2 - x1) / cols, (y2 - y1) / rows
        grid = np.empty((rows, cols, 2), dtype=float)
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = (x1 + (c + .5) * dx, y1 + (r + .5) * dy)
        return grid

    @staticmethod
    def extract_rgb_values(image, centers, x_offset=0, y_offset=0):
        h, w = image.shape[:2]
        ctrs = np.asarray(centers, dtype=float)
        rows, cols = ctrs.shape[:2]
        rgb = [[None]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                x, y = map(int, ctrs[r, c] - (x_offset, y_offset))
                # 取中心和上下左右
                pixels = []
                for dx, dy in [(0,0), (0,-1), (0,1), (-1,0), (1,0)]:
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < w and 0 <= yy < h:
                        pixels.append(image[yy, xx])
                if pixels:
                    b, g, r_ = np.mean(pixels, axis=0)
                    rgb[r][c] = [int(r_), int(g), int(b)]
                else:
                    rgb[r][c] = [0, 0, 0]
        return rgb

    @staticmethod
    def compute_color_matrix(measured, real):
        M = np.array(real, np.float32).T @ np.linalg.pinv(np.array(measured, np.float32).T)
        return M

    @staticmethod
    def apply_color_matrix(rgb_matrix, M):
        arr = np.array(rgb_matrix, np.float32)
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = np.clip(M @ arr[i, j], 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def apply_exposure_contrast(img, exposure, contrast):
        # 曝光和对比度归一化到合理范围
        exp = (exposure - 50) / 50.0  # -1 ~ 1
        ctr = (contrast - 50) / 50.0  # -1 ~ 1
        img = img.astype(np.float32)
        img = img * (1 + exp)         # 曝光调整
        mean = img.mean()
        img = (img - mean) * (1 + ctr) + mean  # 对比度调整
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def apply_brightness_contrast(image, brightness, contrast):
        """
        调整图片的亮度和对比度，接口与UI滑块一致。
        brightness: 0~200, 其中100为无变化（对应beta=0）
        contrast: 0~30, 其中10为无变化（对应alpha=1.0）
        """
        alpha = contrast / 10.0         # Contrast control (1.0 = no change)
        beta = brightness - 100         # Brightness control (0 = no change)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    # ───────────────────────────────────────── swatch-based fine-tune ─────────────────────────────────────────
    def calibrate_rgb_matrix(self, rgb_matrix, calibration_dots, img_full):
        real_rgb = [[255,0,0],[0,255,0],[0,0,255],[255,255,255]]
        measured = [[img_full[int(y), int(x)][::-1] for x,y in calibration_dots][k] for k in range(4)]
        M = self.compute_color_matrix(measured, real_rgb)
        return self.apply_color_matrix(rgb_matrix, M)

    # ───────────────────────────────────────── UI … (identical to your original) ─────────────────────────────────────────
    # draw_ui, update_windows, mouse_callback, run_calibration_ui, calibrate_from_file, calibrate_from_camera
    # ----------------------------------------------------------------
    # --------------------- Calibration UI ----------------------------
    # ----------------------------------------------------------------

    def draw_ui(self, disp):
        """
        Overlay the calibration UI on *disp*:
          • corner handles & index
          • bounding rectangle
          • sample well centres
          • calibration dots for colors
          • instructions & Confirm button
        """
        h, w = disp.shape[:2]

        # ------------------------------------------------------------------
        # 1)  Instructions
        # ------------------------------------------------------------------
        instructions = [
            "Drag corners to define plate region",
            "Drag dots to define calibration colors",
            "Use trackbar: (0=12,1=24,2=48,3=96)",
            "Click 'Confirm' or press 'c' to finalize",
            "Press ESC to cancel"
        ]
        y_off = 25
        for txt in instructions:
            cv2.putText(disp, txt, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_off += 30

        # ------------------------------------------------------------------
        # 2)  Corner handles
        # ------------------------------------------------------------------
        for i, pt in enumerate(self.points):
            cv2.circle(disp, pt, 15, (0, 0, 255), -1)
            cv2.putText(disp, f"{i+1}", (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ------------------------------------------------------------------
        # 3)  Rectangle & sample well centres
        # ------------------------------------------------------------------
        if len(self.points) == 4:
            # lines between corners
            for i in range(4):
                cv2.line(disp, self.points[i],
                         self.points[(i + 1) % 4], (0, 255, 0), 2)

            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)

            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

            plate_val  = cv2.getTrackbarPos("Plate", "Calibration")
            plate_type = self.plate_from_trackbar(plate_val)
            centres    = self.get_well_centers_boxed_grid(rx1, ry1, rx2, ry2,
                                                          plate_type)

            # ── FLATTEN grid to iterate as (cx, cy) pairs ────────────────
            for cx, cy in np.asarray(centres).reshape(-1, 2):
                cv2.circle(disp, (int(cx), int(cy)), 6, (0, 0, 255), -1)

        # ------------------------------------------------------------------
        # 4)  Calibration dots
        # ------------------------------------------------------------------
        calibration_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]  # Red, Green, Blue, White
        for i, (pt, color) in enumerate(zip(self.calibration_dots, calibration_colors)):
            cv2.circle(disp, pt, 10, color, -1)
            cv2.putText(disp, f"C{i+1}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ------------------------------------------------------------------
        # 5)  Confirm button
        # ------------------------------------------------------------------
        self.CONFIRM_BTN_TOPLEFT = (w - self.BTN_WIDTH - 10,
                                    h - self.BTN_HEIGHT - 10)
        bx, by = self.CONFIRM_BTN_TOPLEFT
        cv2.rectangle(disp, (bx, by), (bx + self.BTN_WIDTH, by + self.BTN_HEIGHT),
                      (50, 205, 50), -1)
        cv2.putText(disp, "Confirm", (bx + 10, by + self.BTN_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return disp

    # ───────────────────────────────────────── 关键 UI 刷新 ─────────────────────────────────────────
    def update_windows(self):
        """
        Re-draw the calibration display window with fixed brightness / contrast.
        """
        disp = self.img_copy.copy()
        disp = self.apply_brightness_contrast(disp, 26, 22)  # 固定值
        disp = self.draw_ui(disp)
        cv2.imshow("Calibration", disp)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse event callback for dragging corners or calibration dots.
        Also clamps dragged points to remain within image boundaries.
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
                if np.hypot(x - pt[0], y - pt[1]) < 10:
                    self.dragging_idx = i
                    return

            # Check if near a calibration dot
            for i, pt in enumerate(self.calibration_dots):
                if np.hypot(x - pt[0], y - pt[1]) < 10:
                    self.dragging_calib_idx = i
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                # Dragging a corner
                clamped_x = max(0, min(x, w - 1))
                clamped_y = max(0, min(y, h - 1))
                self.points[self.dragging_idx] = (clamped_x, clamped_y)
                self.update_windows()
            elif self.dragging_calib_idx != -1:
                # Dragging a calibration dot
                clamped_x = max(0, min(x, w - 1))
                clamped_y = max(0, min(y, h - 1))
                self.calibration_dots[self.dragging_calib_idx] = (clamped_x, clamped_y)
                self.update_windows()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1
            self.dragging_calib_idx = -1

    def run_calibration_ui(self, resized_img):
        """
        Launches the calibration window.
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

        # Set default calibration dots (near corners)
        self.calibration_dots = [
            (margin_w + 20, margin_h + 20),  # Red
            (w - margin_w - 20, margin_h + 20),  # Green
            (w - margin_w - 20, h - margin_h - 20),  # Blue
            (margin_w + 20, h - margin_h - 20),  # White/Black
        ]

        self.confirmed = False
        self.dragging_idx = -1
        self.dragging_calib_idx = -1

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        cv2.createTrackbar("Plate", "Calibration", 3, 3, lambda v: self.update_windows())

        self.update_windows()

        print("=== Calibration UI ===")
        print(" - Drag corners to match your plate edges.")
        print(" - Drag dots to define calibration colors.")
        print(" - Trackbar sets plate type (12,24,48,96).")
        print(" - Click 'Confirm' or press 'c' to finalize.")
        print(" - ESC or 3-minute timeout to cancel.\n")

        start_time = time.time()
        TIMEOUT_SECONDS = 180  # 3 minutes

        while True:
            key = cv2.waitKey(20) & 0xFF

            if time.time() - start_time > TIMEOUT_SECONDS:
                print("Calibration timed out. No confirmation within 3 minutes.")
                self.confirmed = False
                break

            if key == 27:  # ESC
                print("Calibration canceled by user (ESC).")
                self.confirmed = False
                break
            elif key == ord('c'):
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
            return None

        return {
            "rectangle": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "plate_type": plate_type,
            "calibration_dots": self.calibration_dots
        }

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

        resized_img, scale = img, 1.0  # No resizing for now

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

    # ───────────────────────────────────────── 后续正式处理 ─────────────────────────────────────────
    def process_image(self, cam_index=0, warmup=10, *, image_path=None,
                      calib_filename="calibration.json", snapshot_file="snapshot.jpg"):

        if image_path is None: image_path = snapshot_file
        self.take_snapshot(cam_index, image_path, warmup_frames=warmup, apply_advanced_wb=True)

        if not os.path.exists(calib_filename):
            print("[INFO] No calibration file – launching UI…")
            calib_data, _, _ = self.calibrate_from_file(image_path, calib_filename)
            if calib_data is None: raise RuntimeError("Calibration aborted")
        else:
            with open(calib_filename) as f: calib_data = json.load(f)

        img = cv2.imread(image_path)  # full resolution, NO RESIZE

        # === 关键：在提取颜色之前统一套用固定亮 / 对 ===
        img = self.apply_brightness_contrast(img, 26, 22)  # 固定值

        rx1, ry1, rx2, ry2 = (calib_data["rectangle"][k] for k in ("x1","y1","x2","y2"))
        centres = self.get_well_centers_boxed_grid(rx1, ry1, rx2, ry2, calib_data["plate_type"])
        raw_rgb = self.extract_rgb_values(img, centres, x_offset=rx1, y_offset=ry1)
        corrected_rgb = self.calibrate_rgb_matrix(raw_rgb, calib_data["calibration_dots"], img)
        return raw_rgb, corrected_rgb



if __name__ == "__main__":
    p = PlateProcessor()
    raw_rgb, corrected_rgb = p.process_image(cam_index=1, warmup=15)
    

    row, col = 0, 2
    print(f"Raw RGB at (0,0) :", raw_rgb[0][0])
    print(f"Raw RGB at (1,0) :", raw_rgb[1][0])
    print(f"Raw RGB at (2,0) :", raw_rgb[2][0])
    print(f"Calibrated RGB at (0,0):", corrected_rgb[0][0])
    print(f"Calibrated RGB at (1,0) :", corrected_rgb[1][0])
    print(f"Calibrated RGB at (2,0) :", corrected_rgb[2][0])
    print(f"Calibrated RGB at (4,0) :", corrected_rgb[4][0])
    print(corrected_rgb[0])
    print(corrected_rgb[1])
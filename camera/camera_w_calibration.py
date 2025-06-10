#!/usr/bin/env python3
# coding: utf-8
"""
camera_color_baseline.py  —  Baseline Color Calibration Pipeline
================================================================
* UI shows 4 draggable dots to define the plate's homography.
* Supports 12 / 24 / 48 / 96 well plates.
* Saves / loads plate corners, plate type, and baseline colors for consistent
  lighting reads.
* Applies a brightness/saturation boost to the final read colors.
* Generates a diagnostic image showing the final adjusted color reads for each well.
"""
# ssh -i C:\Users\shich\.ssh\ot2_ssh_key root@172.26.192.201
from __future__ import annotations
import cv2, time, json, os, argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WIN = "Calibration"            # OpenCV window name

# ═════════════════════════════ PlateProcessor ═════════════════════════════
class PlateProcessor:
    """Camera plate processor.

    Handles camera snapshot, UI calibration for baseline colors, and diagnostic
    image generation. When ``virtual_mode`` is enabled no camera interaction
    occurs and calls to :meth:`process_image` return an all white plate. This
    mirrors the ``OT2Manager``'s virtual mode for easier testing.
    """

    def __init__(self, virtual_mode: bool = False, boost_saturation: bool = False) -> None:
        self.virtual_mode = virtual_mode
        self.boost_saturation = boost_saturation
        # four plate corners
        self.pts: list[tuple[int, int]] = []

        # UI state
        self.drag_idx = -1
        self.img_copy: np.ndarray | None = None
        self.confirmed = False
        self.btnTL: tuple[int, int] | None = None
        self.BW, self.BH = 140, 30          # confirm-button size

    # ───────────────────────────── camera snapshot ────────────────────────
    @staticmethod
    def snapshot(cam: int = 0, path: str = "camera/snapshot.jpg",
                 warm: int = 10, burst: int = 5,
                 res: tuple[int, int] | None = (1600, 1200)) -> str:
        """Capture a denoised snapshot (burst average)."""
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera open failed")
        if res:                          # set resolution before warm-up
            w, h = res
            cap.set(3, w)
            cap.set(4, h)
            time.sleep(0.2)

        print("Warming up camera...")
        for _ in range(warm):            # let exposure settle
            cap.read()
            time.sleep(0.04)

        acc = None
        print("Capturing burst...")
        for _ in range(burst):
            _, frm = cap.read()
            acc = frm.astype(np.float32) if acc is None else acc + frm
            time.sleep(0.02)
        cap.release()

        img = (acc / burst).astype(np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    # ─────────────────────────── misc helper methods ──────────────────────
    @staticmethod
    def plate_from_tb(val: int) -> str:
        """Track-bar value → plate type."""
        return {0: "12", 1: "24", 2: "48", 3: "96"}.get(val, "96")

    @staticmethod
    def well_centers(x1: int, y1: int, x2: int, y2: int,
                     plate: str = "96",
                     quad: list[tuple[int, int]] | None = None) -> np.ndarray:
        """Return an (rows × cols × 2) array of centre coordinates.

        Parameters
        ----------
        x1, y1, x2, y2:
            Bounding rectangle of the plate. Retained for backward
            compatibility. Ignored if ``quad`` is provided.
        plate:
            Plate type ("12", "24", "48" or "96").
        quad:
            Optional list of four corner points (top-left, top-right,
            bottom-right, bottom-left) defining a perspective transform.
        """

        rows, cols = {"12": (8, 12), "24": (4, 6),
                      "48": (6, 8),  "96": (8, 12)}[plate]

        if quad is None:
            dx, dy = (x2 - x1) / cols, (y2 - y1) / rows
            grid = np.empty((rows, cols, 2), float)
            for r in range(rows):
                for c in range(cols):
                    grid[r, c] = (x1 + (c + 0.5) * dx,
                                  y1 + (r + 0.5) * dy)
            return grid

        src = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]],
                       np.float32)
        dst = np.array(quad, np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        xs = np.linspace(0.5, cols - 0.5, cols)
        ys = np.linspace(0.5, rows - 0.5, rows)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        homog = np.concatenate([grid, np.ones((grid.shape[0], 1))], axis=1)
        warped = (H @ homog.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        return warped.reshape(rows, cols, 2)

    # ─────────────────────── per-well trimmed-mean colour ─────────────────

    @staticmethod
    def gaussian_cluster_rgb(img: np.ndarray, centers: np.ndarray,
                             n: int = 80, sigma: float = 4.0,
                             cluster_thresh: float = 10.0) -> list:
        """Sample colours using a Gaussian distribution and return the
        centroid of the largest cluster for each well.

        Parameters
        ----------
        img : np.ndarray
            BGR image from which to sample.
        centers : np.ndarray
            Array of well centre coordinates ``(rows × cols × 2)``.
        n : int
            Number of samples per well.
        sigma : float
            Standard deviation of the Gaussian in pixels.
        cluster_thresh : float
            Euclidean distance threshold for clustering.

        Returns
        -------
        list
            Nested Python lists (rows × cols × 3) of RGB values.
        """
        h, w = img.shape[:2]
        out = []

        def largest_cluster(points: np.ndarray) -> np.ndarray:
            clusters: list[tuple[list[np.ndarray], np.ndarray]] = []
            for p in points:
                assigned = False
                for cl in clusters:
                    if np.linalg.norm(p - cl[1]) <= cluster_thresh:
                        cl[0].append(p)
                        cl[1][:] = np.mean(cl[0], axis=0)
                        assigned = True
                        break
                if not assigned:
                    clusters.append(([p], p.astype(float)))
            if not clusters:
                return np.array([0.0, 0.0, 0.0])
            largest = max(clusters, key=lambda c: len(c[0]))
            return largest[1]

        for row in centers:
            rrow = []
            for cx, cy in row:
                xs = np.random.normal(cx, sigma, n).round().astype(int)
                ys = np.random.normal(cy, sigma, n).round().astype(int)
                xs = np.clip(xs, 0, w - 1)
                ys = np.clip(ys, 0, h - 1)
                samples = img[ys, xs, ::-1].astype(np.float32)  # RGB
                centroid = largest_cluster(samples)
                rrow.append(centroid.tolist())
            out.append(rrow)
        return out

    # ────────────────── Brightness/Saturation Adjustment ──────────────────
    @staticmethod
    def adjust_brightness_saturation(rgb_colors: np.ndarray,
                                     brightness_factor: float = 1.1,
                                     saturation_factor: float = 1.2) -> np.ndarray:
        """
        Adjusts the brightness and saturation of an array of RGB colors using
        the HSV color space.
        
        Parameters
        ----------
        rgb_colors : np.ndarray
            Input array of RGB colors with values in the 0-255 range.
        brightness_factor : float
            Factor to multiply the brightness (Value) by. >1 increases, <1 decreases.
        saturation_factor : float
            Factor to multiply the saturation by. >1 increases, <1 decreases.
            
        Returns
        -------
        np.ndarray
            Array of adjusted RGB colors as float32.
        """
        # Convert to uint8 for HSV conversion, ensuring values are clipped
        img_rgb_u8 = np.clip(rgb_colors, 0, 255).astype(np.uint8)
        
        # Convert RGB to HSV
        img_hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
        
        h, s, v = cv2.split(img_hsv)
        
        # Apply factors to S and V channels, casting to float for multiplication
        # to prevent overflow, then clipping and converting back to uint8.
        s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # Merge the channels and convert back to RGB
        final_hsv = cv2.merge([h, s, v])
        final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        
        return final_rgb.astype(np.float32)

    # ───────────────────────────── UI helpers ─────────────────────────────
    def draw_ui(self, disp: np.ndarray) -> np.ndarray:
        """Overlay instructions, rectangle, sample dots, confirm button."""
        h, w = disp.shape[:2]

        # 1) instructions
        lines = ["Drag 4 plate corners",
                 "Track-bar: plate type",
                 "Press 'c' to confirm, ESC to cancel"]
        for i, txt in enumerate(lines):
            cv2.putText(disp, txt, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2) plate rectangle & well centres
        for i, pt in enumerate(self.pts):
            cv2.circle(disp, pt, 15, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        if len(self.pts) == 4:
            for i in range(4):
                cv2.line(disp, self.pts[i], self.pts[(i + 1) % 4],
                         (0, 255, 0), 2)
            xs = [p[0] for p in self.pts]
            ys = [p[1] for p in self.pts]
            rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)
            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

            plate = self.plate_from_tb(cv2.getTrackbarPos("Plate", WIN))
            for cx, cy in self.well_centers(rx1, ry1, rx2, ry2,
                                            plate,
                                            quad=self.pts).reshape(-1, 2):
                cv2.circle(disp, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # 3) confirm button
        self.btnTL = (w - self.BW - 10, h - self.BH - 10)
        bx, by = self.btnTL
        cv2.rectangle(disp, (bx, by), (bx + self.BW, by + self.BH),
                      (50, 205, 50), -1)
        cv2.putText(disp, "Confirm", (bx + 10, by + self.BH - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return disp

    def update_win(self) -> None:
        cv2.imshow(WIN, self.draw_ui(self.img_copy.copy()))

    # ---------------------------- mouse callback --------------------------
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # confirm button?
            if (self.btnTL and
                self.btnTL[0] <= x <= self.btnTL[0] + self.BW and
                self.btnTL[1] <= y <= self.btnTL[1] + self.BH):
                self.confirmed = True
                return

            # corner?
            for i, pt in enumerate(self.pts):
                if np.hypot(x - pt[0], y - pt[1]) < 12:
                    self.drag_idx = i
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            h, w = self.img_copy.shape[:2]
            if self.drag_idx != -1:
                self.pts[self.drag_idx] = (max(0, min(x, w - 1)),
                                           max(0, min(y, h - 1)))
                self.update_win()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1

    # ----------------------------- calibration UI -------------------------
    def run_ui(self, img: np.ndarray,
               prev: dict | None,
               default_plate: str = "96") -> dict | None:
        """Open the calibration UI; return calibration dict or None if cancel."""
        self.img_copy = img.copy()
        h, w = img.shape[:2]
        mW, mH = int(w * 0.1), int(h * 0.1)

        # preset plate corners
        if prev and "corners" in prev:
            self.pts = [tuple(map(int, p)) for p in prev["corners"]]
        elif prev and "rectangle" in prev:
            r = prev["rectangle"]
            self.pts = [(r["x1"], r["y1"]),
                        (r["x2"], r["y1"]),
                        (r["x2"], r["y2"]),
                        (r["x1"], r["y2"])]
        else:
            self.pts = [(mW, mH), (w - mW, mH),
                        (w - mW, h - mH), (mW, h - mH)]

        self.confirmed = False
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)

        plate_idx = {"12": 0, "24": 1, "48": 2, "96": 3}.get(
            (prev or {}).get("plate_type", default_plate), 3)
        cv2.createTrackbar("Plate", WIN, plate_idx, 3,
                           lambda v: self.update_win())

        self.update_win()
        start = time.time()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or time.time() - start > 180:   # ESC or timeout
                break
            if key == ord('c'):
                self.confirmed = True
                break
            if self.confirmed:
                break

        if not self.confirmed:       # cancelled or timeout
            cv2.destroyWindow(WIN)
            return None

        # read track-bar BEFORE the window is closed!
        plate_idx = cv2.getTrackbarPos("Plate", WIN)
        cv2.destroyWindow(WIN)      # now it is safe to close

        xs=[p[0] for p in self.pts]; ys=[p[1] for p in self.pts]
        rect={"x1":int(min(xs)),"y1":int(min(ys)),
              "x2":int(max(xs)),"y2":int(max(ys))}
        plate = self.plate_from_tb(plate_idx)

        centers = self.well_centers(rect["x1"], rect["y1"], rect["x2"], rect["y2"],
                                    plate, quad=self.pts)
        baseline = self.gaussian_cluster_rgb(img, centers)

        return {
            "rectangle": rect,
            "plate_type": plate,
            "corners": self.pts,
            "baseline_colors": baseline,
        }

    # -------------------------- main processing ---------------------------
    def process_image(self, cam_index: int = 2,
                      snap: str = "camera/snapshot.jpg",
                      calib: str = "camera/calibration.json",
                      force_ui: bool = False,
                      plate_type: str | None = None):
        """Capture and return the adjusted plate colours."""

        if self.virtual_mode:
            cfg = None
            if os.path.exists(calib):
                try:
                    with open(calib) as f:
                        cfg = json.load(f)
                except json.JSONDecodeError:
                    cfg = None

            plate = plate_type or (cfg.get("plate_type") if cfg else "96")
            rows, cols = {"12": (8, 12), "24": (4, 6), "48": (6, 8), "96": (8, 12)}.get(str(plate), (8, 12))
            return np.full((rows, cols, 3), 255.0, dtype=np.float32)

        self.snapshot(cam_index, snap)

        cfg = None
        if os.path.exists(calib):
            with open(calib) as f:
                cfg = json.load(f)

        if plate_type:
            if cfg is None:
                cfg = {"plate_type": plate_type}
            else:
                cfg["plate_type"] = plate_type

        if force_ui or cfg is None or "baseline_colors" not in cfg:
            img_for_ui = cv2.imread(snap)
            if img_for_ui is None:
                raise RuntimeError(f"Failed to read snapshot for UI: {snap}")
            cfg = self.run_ui(img_for_ui, cfg, default_plate=(cfg or {}).get("plate_type", "96"))
            if cfg is None:
                raise RuntimeError("Calibration cancelled")
            with open(calib, "w") as f:
                json.dump(cfg, f, indent=2)

        img = cv2.imread(snap)
        if img is None:
            raise RuntimeError(f"Failed to read snapshot for processing: {snap}")

        # 1) Sample well colours
        r = cfg["rectangle"]
        centers = self.well_centers(r["x1"], r["y1"], r["x2"], r["y2"],
                                    cfg["plate_type"],
                                    quad=cfg.get("corners"))
        raw = self.gaussian_cluster_rgb(img, centers)

        # 2) Apply baseline color correction for consistent lighting
        baseline = cfg.get("baseline_colors")
        if baseline is not None:
            baseline_arr = np.array(baseline, np.float32)
            base_mean = baseline_arr.mean(axis=(0, 1), keepdims=True)
            diff = baseline_arr - base_mean
            raw_bs = np.clip(np.array(raw, np.float32) - diff, 0, None)
        else:
            raw_bs = np.array(raw, np.float32)

        # 3) Apply brightness and saturation adjustment
        if self.boost_saturation:
            # Adjust brightness and saturation
            adjusted_bs = self.adjust_brightness_saturation(raw_bs)
        else:
            # No adjustment, just use the raw baseline colors
            adjusted_bs = raw_bs

        # Save adjusted matrix to a file
        raw_matrix_file = "camera/raw_matrix.json"
        with open(raw_matrix_file, "w") as f:
            json.dump(adjusted_bs.tolist(), f, indent=2)
        print(f"[Saved] Adjusted raw matrix to {raw_matrix_file}")

        # 4) Build diagnostic image
        # The diagnostic image will show the final, adjusted colors.
        disp_colors = np.clip(adjusted_bs, 0, 255).astype(np.uint8)

        marked = img.copy()
        RADIUS = 10

        # Draw left-half-circles for each well with the adjusted color
        for ctr_row, color_row in zip(centers, disp_colors):
            for (cx, cy), rgb_val in zip(ctr_row, color_row):
                bgr_val = tuple(int(v) for v in rgb_val[::-1])
                cv2.ellipse(marked, (int(cx), int(cy)), (RADIUS, RADIUS),
                            0, 90, 270, bgr_val, -1)

        # Save the diagnostic image
        output_file = "camera/output_with_read_colors.jpg"
        cv2.imwrite(output_file, marked)
        print(f"[Saved] {output_file}")
        
        return adjusted_bs

# ═══════════════════════════════════ CLI ══════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture calibration data and read plate colors.")
    parser.add_argument("--cam-index", type=int, default=2, help="Camera index")
    parser.add_argument("--force-ui", action="store_true", help="Always show calibration UI")
    parser.add_argument("--plate-type", choices=["12", "24", "48", "96"], help="Plate type/well count")
    parser.add_argument("--robot-number", type=int,
                        help="Load OT2 connection details from secret/OT_<num>/info.json")
    args = parser.parse_args()

    robot = None
    if args.robot_number is not None:
        info_path = f"secret/OT_{args.robot_number}/info.json"
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception as e:
            raise SystemExit(f"Failed to read {info_path}: {e}")

        local_ip = info.get("local_ip")
        local_pw = info.get("local_password")
        local_pw = None if local_pw in (None, "None") else local_pw
        remote_ip = info.get("remote_ip")
        remote_pw = info.get("remote_password")
        remote_pw = None if remote_pw in (None, "None") else remote_pw
        local_key = f"secret/OT_{args.robot_number}/ot2_ssh_key"
        remote_key = f"secret/OT_{args.robot_number}/ot2_ssh_key_remote"

        from color_matching.robot.ot2_utils import OT2Manager
        try:
            raise NotImplementedError("temp bypass to force remote")
            robot = OT2Manager(hostname=local_ip,
                               username="root",
                               password=local_pw,
                               key_filename=local_key,
                               bypass_startup_key=True)
            print("Connected to OT2 locally.")
        except Exception as e:
            print(f"Local connection failed: {e}. Trying remote...")
            robot = OT2Manager(hostname=remote_ip,
                               username="root",
                               password=remote_pw,
                               key_filename=remote_key,
                               bypass_startup_key=True)
            print("Connected to OT2 remotely.")

        robot.add_turn_on_lights_action()
        robot.execute_actions_on_remote()

        plate_colors = PlateProcessor().process_image(
            cam_index=args.cam_index,
            force_ui=args.force_ui,
            plate_type=args.plate_type,
            calib=f"secret/OT_{args.robot_number}/calibration.json"
        )

        robot.add_turn_off_lights_action()
        robot.add_close_action()
        robot.execute_actions_on_remote()
    else:
        plate_colors = PlateProcessor().process_image(
            cam_index=args.cam_index,
            force_ui=args.force_ui,
            plate_type=args.plate_type,
        )

    print("Processing complete.")
    print("First row of read RGB values:", plate_colors[0])
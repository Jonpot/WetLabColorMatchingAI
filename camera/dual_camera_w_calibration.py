#!/usr/bin/env python3
# coding: utf-8
"""
dual_camera_color_baseline.py — Dual-Plate Baseline Color Calibration Pipeline
================================================================================
* UI shows 8 draggable dots to define the homography for two separate plates.
* Supports 12 / 24 / 48 / 96 well plates (assumes both plates are the same type).
* Saves / loads plate corners, plate type, and baseline colors for both plates.
* Applies a brightness/saturation boost to the final read colors.
* Generates a single diagnostic image showing the final adjusted color reads for
  both plates.
* Returns a dictionary with color data for each plate.
"""
from __future__ import annotations
import cv2, time, json, os, argparse
from .camera_stream import get_stream
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WIN = "Dual Plate Calibration"      # OpenCV window name

# ═════════════════════════════ DualPlateProcessor ═════════════════════════════
class DualPlateProcessor:
    """Camera two-plate processor.

    Handles camera snapshot, UI calibration for two plates' baseline colors,
    and diagnostic image generation.
    """

    def __init__(self, virtual_mode: bool = False, boost_saturation: bool = False) -> None:
        self.virtual_mode = virtual_mode
        self.boost_saturation = boost_saturation
        # Store corners for two plates
        self.pts: dict[str, list[tuple[int, int]]] = {'plate_1': [], 'plate_2': []}

        # UI state for dragging
        self.drag_key: str | None = None  # 'plate_1' or 'plate_2'
        self.drag_idx: int = -1
        
        self.img_copy: np.ndarray | None = None
        self.confirmed = False
        self.btnTL: tuple[int, int] | None = None
        self.BW, self.BH = 140, 30          # confirm-button size

    # ───────────────────────────── camera snapshot ────────────────────────
    @staticmethod
    def snapshot(cam: int = 0, path: str = "camera/snapshot.jpg",
                 warm: int = 10, burst: int = 5,
                 res: tuple[int, int] | None = (1600, 1200)) -> str:
        """Return the latest frame captured by a background thread."""
        stream = get_stream(cam_index=cam, res=res, warm=warm)
        img = stream.read()
        if img is None:
            raise RuntimeError("No frame captured")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    # ─────────────────────────── misc helper methods ──────────────────────
    @staticmethod
    def plate_from_tb(val: int) -> str:
        """Track-bar value → plate type."""
        return {0: "12", 1: "24", 2: "48", 3: "96"}.get(val, "96")

    @staticmethod
    def well_centers(quad: list[tuple[int, int]] = None, plate_type: str = "96") -> np.ndarray:
        """Return an (rows × cols × 2) array of centre coordinates."""
        rows, cols = {"12": (8, 12), "24": (4, 6),
                      "48": (6, 8),  "96": (8, 12)}[plate_type]

        src = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], np.float32)
        dst = np.array(quad, np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        xs = np.linspace(0.5, cols - 0.5, cols)
        ys = np.linspace(0.5, rows - 0.5, rows)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        homog = np.concatenate([grid, np.ones((grid.shape[0], 1))], axis=1)
        warped = (H @ homog.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        return warped.reshape(rows, cols, 2)

    @staticmethod
    def gaussian_cluster_rgb(img: np.ndarray, centers: np.ndarray,
                             n: int = 80, sigma: float = 4.0,
                             cluster_thresh: float = 10.0) -> list:
        """Sample colours using a Gaussian distribution and return the
        centroid of the largest cluster for each well."""
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
            if not clusters: return np.array([0.0, 0.0, 0.0])
            return max(clusters, key=lambda c: len(c[0]))[1]

        for row in centers:
            rrow = []
            for cx, cy in row:
                xs = np.clip(np.random.normal(cx, sigma, n).round().astype(int), 0, w - 1)
                ys = np.clip(np.random.normal(cy, sigma, n).round().astype(int), 0, h - 1)
                samples = img[ys, xs, ::-1].astype(np.float32)  # BGR -> RGB
                rrow.append(largest_cluster(samples).tolist())
            out.append(rrow)
        return out

    # ────────────────── Brightness/Saturation Adjustment ──────────────────
    @staticmethod
    def adjust_brightness_saturation(rgb_colors: np.ndarray,
                                     brightness_factor: float = 1.1,
                                     saturation_factor: float = 1.2) -> np.ndarray:
        """Adjusts the brightness and saturation of an array of RGB colors."""
        img_rgb_u8 = np.clip(rgb_colors, 0, 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge([h, s, v])
        final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return final_rgb.astype(np.float32)

    # ───────────────────────────── UI helpers ─────────────────────────────
    def draw_ui(self, disp: np.ndarray) -> np.ndarray:
        """Overlay instructions and calibration points for two plates."""
        h, w = disp.shape[:2]

        lines = ["Drag corners for 2 plates (1-4 Red, 5-8 Green)",
                 "Track-bar: Plate Type (applies to both plates)",
                 "Press 'c' to confirm, ESC to cancel"]
        for i, txt in enumerate(lines):
            cv2.putText(disp, txt, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        plate_colors = {'plate_1': (0, 0, 255), 'plate_2': (0, 255, 0)}  # Red, Green
        plate_type = self.plate_from_tb(cv2.getTrackbarPos("Plate Type", WIN))

        for key, corners in self.pts.items():
            color = plate_colors[key]
            start_idx = 1 if key == 'plate_1' else 5
            for i, pt in enumerate(corners):
                cv2.circle(disp, pt, 15, color, -1)
                cv2.putText(disp, str(start_idx + i), (pt[0] - 5, pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if len(corners) == 4:
                for i in range(4):
                    cv2.line(disp, corners[i], corners[(i + 1) % 4], color, 2)
                for cx, cy in self.well_centers(corners, plate_type).reshape(-1, 2):
                    cv2.circle(disp, (int(cx), int(cy)), 4, color, -1)

        self.btnTL = (w - self.BW - 10, h - self.BH - 10)
        bx, by = self.btnTL
        cv2.rectangle(disp, (bx, by), (bx + self.BW, by + self.BH), (50, 205, 50), -1)
        cv2.putText(disp, "Confirm", (bx + 10, by + self.BH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return disp

    def update_win(self) -> None:
        cv2.imshow(WIN, self.draw_ui(self.img_copy.copy()))

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.btnTL and self.btnTL[0] <= x <= self.btnTL[0] + self.BW and
                    self.btnTL[1] <= y <= self.btnTL[1] + self.BH):
                self.confirmed = True
                return

            for key, corners in self.pts.items():
                for i, pt in enumerate(corners):
                    if np.hypot(x - pt[0], y - pt[1]) < 15:
                        self.drag_key = key
                        self.drag_idx = i
                        return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_key and self.drag_idx != -1:
                h, w = self.img_copy.shape[:2]
                self.pts[self.drag_key][self.drag_idx] = (max(0, min(x, w-1)), max(0, min(y, h-1)))
                self.update_win()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_key = None
            self.drag_idx = -1

    # ----------------------------- calibration UI -------------------------
    def run_ui(self, img: np.ndarray, prev: dict | None) -> dict | None:
        """Open the calibration UI for two plates."""
        self.img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Set initial points for two plates if none are loaded
        for key, default_corners in [
            ('plate_1', [(int(w*0.1), int(h*0.1)), (int(w*0.4), int(h*0.1)),
                         (int(w*0.4), int(h*0.9)), (int(w*0.1), int(h*0.9))]),
            ('plate_2', [(int(w*0.6), int(h*0.1)), (int(w*0.9), int(h*0.1)),
                         (int(w*0.9), int(h*0.9)), (int(w*0.6), int(h*0.9))])
        ]:
            if prev and prev.get(key) and 'corners' in prev[key]:
                 self.pts[key] = [tuple(map(int, p)) for p in prev[key]['corners']]
            else:
                 self.pts[key] = default_corners

        self.confirmed = False
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)
        
        plate_idx = {"12": 0, "24": 1, "48": 2, "96": 3}.get((prev or {}).get("plate_type", "96"), 3)
        cv2.createTrackbar("Plate Type", WIN, plate_idx, 3, lambda v: self.update_win())
        
        self.update_win()
        start = time.time()
        while not self.confirmed and not (cv2.waitKey(20) & 0xFF == 27 or time.time() - start > 180):
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break # Exit if window closed
        
        if not self.confirmed:
            cv2.destroyWindow(WIN)
            return None

        plate_type = self.plate_from_tb(cv2.getTrackbarPos("Plate Type", WIN))
        cv2.destroyWindow(WIN)

        final_calib = {"plate_type": plate_type}
        for key, corners in self.pts.items():
            centers = self.well_centers(corners, plate_type)
            final_calib[key] = {
                "corners": corners,
                "baseline_colors": self.gaussian_cluster_rgb(img, centers),
            }
        return final_calib

    # -------------------------- main processing ---------------------------
    def process_image(self, cam_index: int = 2,
                      snap: str = "camera/snapshot.jpg",
                      calib: str = "camera/dual_calibration.json",
                      force_ui: bool = False,
                      plate_type_override: str | None = None) -> dict[str, np.ndarray]:
        """Capture and return the adjusted colours for two plates."""
        
        self.snapshot(cam_index, snap)

        cfg = None
        if os.path.exists(calib):
            with open(calib) as f: cfg = json.load(f)

        if plate_type_override:
            if cfg is None: cfg = {}
            cfg["plate_type"] = plate_type_override

        if force_ui or cfg is None or "plate_1" not in cfg or "plate_2" not in cfg:
            img_for_ui = cv2.imread(snap)
            if img_for_ui is None: raise RuntimeError(f"Failed to read snapshot for UI: {snap}")
            cfg = self.run_ui(img_for_ui, cfg)
            if cfg is None: raise RuntimeError("Calibration cancelled")
            with open(calib, "w") as f: json.dump(cfg, f, indent=2)

        img = cv2.imread(snap)
        if img is None: raise RuntimeError(f"Failed to read snapshot for processing: {snap}")

        results = {}
        plate_type = cfg.get("plate_type", "96")
        
        for key in ['plate_1', 'plate_2']:
            plate_cfg = cfg[key]
            centers = self.well_centers(plate_cfg["corners"], plate_type)
            raw = self.gaussian_cluster_rgb(img, centers)

            baseline = plate_cfg.get("baseline_colors")
            baseline_arr = np.array(baseline, np.float32)
            base_mean = baseline_arr.mean(axis=(0, 1), keepdims=True)
            diff = baseline_arr - base_mean
            raw_bs = np.clip(np.array(raw, np.float32) - diff, 0, None)
            
            results[key] = self.adjust_brightness_saturation(raw_bs) if self.boost_saturation else raw_bs

        # Save results to a single JSON file
        results_for_json = {k: v.tolist() for k, v in results.items()}
        with open("camera/dual_raw_matrix.json", "w") as f:
            json.dump(results_for_json, f, indent=2)
        print("[Saved] Adjusted dual raw matrix to camera/dual_raw_matrix.json")

        # Build and save diagnostic image
        marked = img.copy()
        RADIUS = 10
        for key, adjusted_colors in results.items():
            centers = self.well_centers(cfg[key]["corners"], plate_type)
            disp_colors = np.clip(adjusted_colors, 0, 255).astype(np.uint8)
            for ctr_row, color_row in zip(centers, disp_colors):
                for (cx, cy), rgb_val in zip(ctr_row, color_row):
                    bgr_val = tuple(int(v) for v in rgb_val[::-1])
                    cv2.ellipse(marked, (int(cx), int(cy)), (RADIUS, RADIUS), 0, 90, 270, bgr_val, -1)
        
        output_file = "camera/dual_output_with_read_colors.jpg"
        cv2.imwrite(output_file, marked)
        print(f"[Saved] {output_file}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture calibration data and read plate colors.")
    parser.add_argument("--cam-index", type=int, default=None, help="Camera index")
    parser.add_argument("--force-ui", action="store_true", help="Always show calibration UI")
    parser.add_argument("--plate-type", choices=["12", "24", "48", "96"], help="Plate type/well count")
    parser.add_argument("--robot-number", type=int,
                        help="Load OT2 connection details from secret/OT_<num>/info.json")
    parser.add_argument("--force-remote", type=bool, default=False,
                        help="Force remote connection even if local connection is available")
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

        if args.cam_index is None:
            args.cam_index = info.get("cam_index", 2)

        from color_matching.robot.ot2_utils import OT2Manager
        try:
            if args.force_remote or not local_ip or not local_pw or not Path(local_key).exists():
                raise RuntimeError("Forced remote connection")
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

        plate_colors = DualPlateProcessor().process_image(
            cam_index=args.cam_index,
            force_ui=args.force_ui,
            plate_type_override=args.plate_type,
            calib=f"secret/OT_{args.robot_number}/dual_calibration.json"
        )

        robot.add_turn_off_lights_action()
        robot.add_close_action()
        robot.execute_actions_on_remote()
    else:
        plate_colors = DualPlateProcessor().process_image(
            cam_index=args.cam_index,
            force_ui=args.force_ui,
            plate_type=args.plate_type,
        )

    print("Processing complete.")
    print("First row of Plate1 read RGB values:", plate_colors['plate_1'][0])
    print("Second row of Plate2 read RGB values:", plate_colors['plate_2'][0])
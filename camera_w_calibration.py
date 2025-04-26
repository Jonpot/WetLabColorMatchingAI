#!/usr/bin/env python3
# coding: utf-8
"""
camera_color24.py  ―  24-patch ColorChecker calibration pipeline
================================================================
* UI shows 24 draggable dots – drop them onto the chart.
* Supports 12 / 24 / 48 / 96 well plates (5 × 5 trimmed-mean sampling).
* Saves / loads rectangle, 24 dots, plate type, brightness, contrast.
* Learns a 3 × 10 **root-polynomial colour-correction matrix** each run.
* Generates a diagnostic image:
    ├ each well centre:   left-half = raw,    right-half = corrected
    └ each chart patch:   left-half = raw,    right-half = corrected,
                          right-shifted solid circle = Macbeth reference
"""
# ssh -i C:\Users\shich\.ssh\ot2_ssh_key root@172.26.192.201  
from __future__ import annotations
import cv2, time, json, os, argparse
import numpy as np
from pathlib import Path

WIN = "Calibration"            # OpenCV window name

# ───────────────────────────────── 24-patch Macbeth reference colours ─────
MACBETH_24_BGR = [
    ( 68,  82,115),(130,150,194),(157,122, 98),( 67,108, 87),
    (177,128,133),(170,189,103),( 44,126,214),(166, 91, 80),
    ( 99,  90,193),(108, 60, 94),( 64,188,157),( 46,163,224),
    (150, 61, 56),( 73,148, 70),( 60, 54,175),( 31,199,231),
    (149, 86,187),(161,133,  8),(242,243,242),(200,200,200),
    (160,160,160),(121,122,122),( 85, 85, 85),( 52, 52, 52)]
MACBETH_24_RGB = np.array([[b, g, r] for r, g, b in MACBETH_24_BGR],
                          np.float32)            # shape (24, 3)

# ─────────────────────────── sRGB ↔ linear helpers ────────────────────────
def srgb2lin(s: np.ndarray) -> np.ndarray:
    """8-bit sRGB → linear 0-1."""
    s = s / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)

def lin2srgb(l: np.ndarray) -> np.ndarray:
    """linear 0-1 → 8-bit sRGB."""
    l = np.clip(l, 0.0, 1.0)          # ← add this line
    s = np.where(l <= 0.0031308,
                 12.92 * l,
                 1.055 * np.power(l, 1/2.4) - 0.055)
    return np.clip(s * 255, 0, 255)

# ═════════════════════════════ PlateProcessor ═════════════════════════════
class PlateProcessor:
    """
    Handles camera snapshot, UI calibration, colour correction, and diagnostic
    image generation.
    """

    def __init__(self) -> None:
        # four plate corners and 24 colour-chart points
        self.pts:  list[tuple[int, int]] = []
        self.cpts: list[tuple[int, int]] = []

        # UI state
        self.drag_idx = self.drag_cidx = -1
        self.img_copy: np.ndarray | None = None
        self.confirmed = False
        self.btnTL: tuple[int, int] | None = None
        self.BW, self.BH = 140, 30             # confirm-button size

    # ───────────────────────────── camera snapshot ────────────────────────
    @staticmethod
    def snapshot(cam: int = 0, path: str = "snapshot.jpg",
                 warm: int = 10, burst: int = 5,
                 res: tuple[int, int] | None = (1600, 1200)) -> str:
        """Capture a denoised snapshot (burst average)."""
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera open failed")
        if res:                        # set resolution before warm-up
            w, h = res
            cap.set(3, w)
            cap.set(4, h)
            time.sleep(0.2)

        for _ in range(warm):          # let exposure settle
            cap.read()
            time.sleep(0.04)

        acc = None
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
                     plate: str = "96") -> np.ndarray:
        """Return an (rows × cols × 2) array of centre coordinates."""
        rows, cols = {"12": (8, 12), "24": (4, 6),
                      "48": (6, 8),  "96": (8, 12)}[plate]
        dx, dy = (x2 - x1) / cols, (y2 - y1) / rows
        grid = np.empty((rows, cols, 2), float)
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = (x1 + (c + 0.5) * dx,
                              y1 + (r + 0.5) * dy)
        return grid

    # ─────────────────────── per-well trimmed-mean colour ─────────────────
    @staticmethod
    def avg_rgb(img: np.ndarray, centers: np.ndarray,
                win: int = 11, trim: float = 0.1) -> list:
        """
        For each centre, take a `win`×`win` patch, drop upper/lower
        `trim` fraction, return RGB mean. Returns nested Python lists for
        JSON serialisability.
        """
        out = []
        h, w = img.shape[:2]
        half = win // 2
        k_trim = int((win * win) * trim)

        for row in centers:
            rrow = []
            for cx, cy in row:
                x, y = int(cx), int(cy)
                x1, x2 = max(0, x - half), min(w, x + half + 1)
                y1, y2 = max(0, y - half), min(h, y + half + 1)
                patch = img[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)

                if k_trim:                             # trimmed mean
                    patch = np.sort(patch, axis=0)[k_trim:-k_trim] \
                            if patch.shape[0] > 2 * k_trim else patch
                rrow.append(patch.mean(axis=0)[::-1])  # BGR→RGB
            out.append(rrow)
        return out

    # ───────────────────── root-polynomial colour correction ──────────────
    # 10-term basis: R, G, B, √RG, √RB, √GB, R², G², B², 1
    @staticmethod
    def _poly_terms(lin: np.ndarray) -> np.ndarray:
        r, g, b = lin.T
        return np.stack([r, g, b,
                         np.sqrt(r*g), np.sqrt(r*b), np.sqrt(g*b),
                         r*r, g*g, b*b,
                         np.ones_like(r)], axis=1)       # N × 10

    @staticmethod
    def fit_rpcc(meas_rgb8: np.ndarray,
                 ref_rgb8: np.ndarray) -> np.ndarray:
        """Return 3 × 10 root-polynomial CCM."""
        X = PlateProcessor._poly_terms(srgb2lin(meas_rgb8))
        Y = srgb2lin(ref_rgb8)             # N × 3
        return Y.T @ np.linalg.pinv(X.T)   # 3 × 10

    @staticmethod
    def apply_rpcc(rgb8: np.ndarray, M10: np.ndarray) -> np.ndarray:
        """Apply 3 × 10 CCM to an RGB array of any shape."""
        lin = srgb2lin(rgb8.astype(np.float32).reshape(-1, 3))
        corr_lin = (M10 @ PlateProcessor._poly_terms(lin).T).T
        return lin2srgb(corr_lin).reshape(rgb8.shape) \
                                 .clip(0, 255).astype(np.uint8)

    # ───────────────────────────── UI helpers ─────────────────────────────
    def draw_ui(self, disp: np.ndarray) -> np.ndarray:
        """Overlay instructions, rectangle, sample dots, confirm button."""
        h, w = disp.shape[:2]

        # 1) instructions
        lines = ["Drag 4 plate corners",
                 "Drag 24 coloured dots to chart",
                 "Track-bar: plate type   (press 'c' to confirm, ESC to cancel)"]
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
                                            plate).reshape(-1, 2):
                cv2.circle(disp, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # 3) 24 chart dots
        for i, pt in enumerate(self.cpts):
            cv2.circle(disp, pt, 10, MACBETH_24_BGR[i], -1)
            cv2.putText(disp, f"{i+1}", (pt[0] - 5, pt[1] + 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 0), 2)

        # 4) confirm button
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
            # colour dot?
            for i, pt in enumerate(self.cpts):
                if np.hypot(x - pt[0], y - pt[1]) < 12:
                    self.drag_cidx = i
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            h, w = self.img_copy.shape[:2]
            if self.drag_idx != -1:
                self.pts[self.drag_idx] = (max(0, min(x, w - 1)),
                                           max(0, min(y, h - 1)))
                self.update_win()
            elif self.drag_cidx != -1:
                self.cpts[self.drag_cidx] = (max(0, min(x, w - 1)),
                                             max(0, min(y, h - 1)))
                self.update_win()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = self.drag_cidx = -1

    # ----------------------------- calibration UI -------------------------
    def run_ui(self, img: np.ndarray,
               prev: dict | None) -> dict | None:
        """Open the calibration UI; return calibration dict or None if cancel."""
        self.img_copy = img.copy()
        h, w = img.shape[:2]
        mW, mH = int(w * 0.1), int(h * 0.1)

        # preset rectangle
        if prev and "rectangle" in prev:
            r = prev["rectangle"]
            self.pts = [(r["x1"], r["y1"]),
                        (r["x2"], r["y1"]),
                        (r["x2"], r["y2"]),
                        (r["x1"], r["y2"])]
        else:
            self.pts = [(mW, mH), (w - mW, mH),
                        (w - mW, h - mH), (mW, h - mH)]

        # preset 24 dots
        if prev and len(prev.get("calibration_dots", [])) == 24:
            self.cpts = [tuple(map(int, p)) for p in prev["calibration_dots"]]
        else:
            gx = np.linspace(mW + 30, w - mW - 30, 6)
            gy = np.linspace(mH + 30, h - mH - 30, 4)
            self.cpts = [(int(x), int(y)) for y in gy for x in gx]

        self.confirmed = False
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)

        plate_idx = {"12": 0, "24": 1, "48": 2, "96": 3}.get(
            (prev or {}).get("plate_type", "96"), 3)
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

        if not self.confirmed:      # cancelled or timeout
            cv2.destroyWindow(WIN)
            return None

        # read track-bar BEFORE the window is closed!
        plate_idx = cv2.getTrackbarPos("Plate", WIN)
        cv2.destroyWindow(WIN)      # now it is safe to close

        xs=[p[0] for p in self.pts]; ys=[p[1] for p in self.pts]
        rect={"x1":int(min(xs)),"y1":int(min(ys)),
              "x2":int(max(xs)),"y2":int(max(ys))}
        plate = self.plate_from_tb(plate_idx)
        
        return {"rectangle": rect,
                "plate_type": plate,
                "calibration_dots": self.cpts}

    # -------------------------- main processing ---------------------------
    def process_image(self, cam_index: int = 0,
                      snap: str = "snapshot.jpg",
                      calib: str = "calibration.json",
                      force_ui: bool = False):
        """Main workflow: snapshot → calibration → corrected RGB matrix."""
        self.snapshot(cam_index, snap)

        cfg = None
        if os.path.exists(calib):
            with open(calib) as f:
                cfg = json.load(f)

        if force_ui or cfg is None:
            cfg = self.run_ui(cv2.imread(snap), cfg)
            if cfg is None:
                raise RuntimeError("Calibration cancelled")
            with open(calib, "w") as f:
                json.dump(cfg, f, indent=2)

        img = cv2.imread(snap)

        # 1) sample well colours
        r = cfg["rectangle"]
        centres = self.well_centers(r["x1"], r["y1"], r["x2"], r["y2"],
                                    cfg["plate_type"])
        raw = self.avg_rgb(img, centres)             # nested lists (rows × cols)

        # 2) build RPCC from 24 patches
        dots = np.asarray(cfg["calibration_dots"], int)
        meas_raw = img[dots[:, 1], dots[:, 0], ::-1].astype(np.float32)  # N×3 RGB
        M10 = self.fit_rpcc(meas_raw, MACBETH_24_RGB)
        meas_corr = self.apply_rpcc(meas_raw, M10)
        corr = self.apply_rpcc(np.array(raw, np.float32), M10)

        # Save corrected matrix to a file
        corrected_matrix_file = "corrected_matrix.json"
        with open(corrected_matrix_file, "w") as f:
            json.dump(corr.tolist(), f, indent=2)
        print(f"[Saved] Corrected matrix to {corrected_matrix_file}")

        # 3) build diagnostic image
        alpha = cfg.get("contrast", 10) / 10.0
        beta  = cfg.get("brightness", 100) - 100
        raw_adj = np.clip(np.array(raw) * alpha + beta,
                          0, 255).astype(np.uint8)

        marked = img.copy()
        RADIUS = 10

        # (a) wells: half raw / half corrected
        for ctr_row, raw_row, cor_row in zip(centres, raw_adj, corr):
            for (cx, cy), rgb_r, rgb_c in zip(ctr_row, raw_row, cor_row):
                cv2.circle(marked, (int(cx), int(cy)), RADIUS,
                           tuple(int(v) for v in rgb_c[::-1]), -1)
                cv2.ellipse(marked, (int(cx), int(cy)), (RADIUS, RADIUS),
                            0, 90, 270,
                            tuple(int(v) for v in rgb_r[::-1]), -1)

        # (b) 24-patch dots: half raw / half corr + reference circle
        SHIFT = RADIUS + 10
        for i, ((x, y), rgb_r, rgb_c) in enumerate(zip(dots,
                                                       meas_raw,
                                                       meas_corr)):
            centre = (int(x), int(y))
            bgr_r  = tuple(int(v) for v in rgb_r[::-1]) #raw color
            bgr_c  = tuple(int(v) for v in rgb_c[::-1]) #correction color
            bgr_ref = MACBETH_24_BGR[i] #reference color

            # half-circle
            cv2.circle(marked, centre, RADIUS, bgr_c, -1)
            cv2.ellipse(marked, centre, (RADIUS, RADIUS),
                        0, 90, 270, bgr_r, -1)
            # reference solid circle
            cv2.circle(marked,
                       (centre[0] + SHIFT, centre[1]),
                       RADIUS, bgr_ref, -1)

        cv2.imwrite("output_with_centers_corr.jpg", marked)
        print("[Saved] output_with_centers_corr.jpg")
        return corr

# ═══════════════════════════════════ CLI ══════════════════════════════════
if __name__ == "__main__":
    from ot2_utils import OT2Manager
    # robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None)
    # robot.add_turn_on_lights_action()
    # robot.execute_actions_on_remote()

    corr = PlateProcessor().process_image(cam_index=1, force_ui=True)

    # robot.add_turn_off_lights_action()
    # robot.add_close_action()
    # robot.execute_actions_on_remote()
    # print("First corrected RGB (row 0):", corr[0])

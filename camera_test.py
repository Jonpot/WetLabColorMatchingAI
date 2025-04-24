#!/usr/bin/env python3
# camera_test.py – enumerate & live-tweak webcam properties (robust version)

import cv2, sys, inspect

###############################################################################
# 1. collect every cv2.CAP_PROP_* except read-only pseudo-props
###############################################################################
READ_ONLY = {cv2.CAP_PROP_BACKEND, cv2.CAP_PROP_SETTINGS}
CAPS = {
    n: v for n, v in inspect.getmembers(cv2)
    if n.startswith("CAP_PROP_") and isinstance(v, int) and v not in READ_ONLY
}
CAPS = dict(sorted(CAPS.items(), key=lambda kv: kv[1]))

###############################################################################
# 2. open camera
###############################################################################
cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit(f"❌  Cannot open camera index {cam_idx}")

print(f"\n=== Camera {cam_idx} | OpenCV {cv2.__version__} | backend={cap.getBackendName()} ===")
hdr = f"{'Property':38}  {'Value':>12}  {'Writable':>9}"
print(hdr); print("-" * len(hdr))

adjustables = {}      # {name: (propId, lo, hi)}

###############################################################################
# 3. probe every property
###############################################################################
for name, pid in CAPS.items():
    val = cap.get(pid)
    if val == -1:
        print(f"{name:38}  {'N/A':>12}  {'—':>9}")
        continue

    writable = False
    try:
        # try nudging the value by ±10 %
        test = val * 1.1 if val != 0 else 1.0
        test = max(min(test, 1e6), -1e6)
        if cap.set(pid, test):
            echoed = cap.get(pid)
            writable = abs(echoed - val) >= max(1.0, 0.01 * abs(val))
        cap.set(pid, val)  # restore
    except cv2.error:      # driver threw “can’t set read-only”
        writable = False

    flag = "✓" if writable else "RO"
    print(f"{name:38}  {val:12.3f}  {flag:>9}")

    if writable and len(adjustables) < 12:
        lo = val * 0.2
        hi = val * 5.0
        if val == 0:       # avoid hi==lo → division-by-zero
            lo, hi = 0.0, 60.0
        if hi - lo >= 1.0:
            adjustables[name] = (pid, lo, hi)

print(f"\nAdjustable controls found: {len(adjustables)} (sliders for ≤12)\n")

###############################################################################
# 4. live preview with sliders
###############################################################################
cv2.namedWindow("live", cv2.WINDOW_NORMAL)

def make_slider(prop, pid, lo, hi):
    def on_change(pos):
        try:
            cap.set(pid, lo + (hi - lo) * (pos / 1000.0))
        except cv2.error:
            pass  # ignore driver errors
    init = int((cap.get(pid) - lo) / (hi - lo) * 1000) if hi > lo else 0
    cv2.createTrackbar(prop, "live", max(0, min(init, 1000)), 1000, on_change)

for n, (pid, lo, hi) in adjustables.items():
    make_slider(n, pid, lo, hi)

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  frame grab failed"); break
    cv2.imshow("live", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()

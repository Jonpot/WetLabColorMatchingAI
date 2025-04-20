# PlateProcessor API Documentation

## Overview

The `PlateProcessor` class provides a high-level interface for **calibrating** and **extracting color data** from plate images. It offers an **interactive calibration window** where you can drag corner points, select a plate type (12‑, 24‑, 48‑, or 96‑well), and save the resulting configuration to a JSON file for subsequent automated processing. It also includes **camera utilities** for taking snapshots and listing available cameras.

### Key Features

- **Interactive Corner‑Dragging UI**  
  Manually calibrate the plate region by dragging four corner points. The code **clamps** these corners so they always stay inside the image boundaries.  
- **3‑Minute Timeout**  
  If you do not confirm within 3 minutes, the UI automatically exits and reports that calibration was canceled.  
- **Automatic JSON Storage**  
  Calibration data are persisted in a `.json` file, so repeated calibrations are unnecessary unless you explicitly choose to re‑calibrate.  
- **Well‑Center Computation**  
  Once a plate type is chosen, the class automatically computes the (x, y) center coordinates for each well.  
- **Color Extraction**  
  Collects RGB values at each well center, optionally averaging a small neighborhood for noise reduction.  
- **Camera Utilities**  
  Helper functions let you list available cameras and capture snapshots via DirectShow.  
- **One‑Step Capture & Processing**  
  The revamped `process_image` method **always captures a fresh snapshot**, creates calibration data on first use, and then returns a `(3 × N)` RGB matrix in a single call.

---

## Requirements

- **Python 3.x**  
- **OpenCV (cv2)** – `pip install opencv-python`  
- **NumPy** – `pip install numpy`  
- **Built‑in modules:** `json`, `os`, `time`  
- **Optional:** SciPy (for `compute_rgb_statistics`) – `pip install scipy`

---

## Class: `PlateProcessor`

### Initialization

```python
processor = PlateProcessor()
```

This prepares the interactive calibration logic (corner points, mouse callbacks, confirmation flag), clamps dragged corners within image boundaries, and enforces the 3‑minute UI timeout.

---

## Methods

### Camera & Utility Methods

<details>
<summary><code>list_cameras(max_tested=10) -> List[int]</code></summary>

Checks camera indices `0 … max_tested‑1` using DirectShow and returns the usable indices.

```python
valid_cams = PlateProcessor.list_cameras()
print(valid_cams)
```
</details>

<details>
<summary><code>take_snapshot(cam_index=0, save_path="snapshot.jpg", warmup_frames=10, properties=None) -> str</code></summary>

Captures one image from the specified camera. Optional `properties` lets you adjust resolution or other `cv2.CAP_PROP_*` settings.

```python
snap = PlateProcessor.take_snapshot(cam_index=1, save_path="my_photo.jpg")
```
</details>

---

### Calibration Methods

<details>
<summary><code>calibrate_from_file(image_path, calib_filename="calibration.json") -> Tuple[dict, np.ndarray, float]</code></summary>

Performs calibration using an existing image. If `calibration.json` already exists, you can reuse or overwrite it; otherwise an interactive window appears. The UI times out after 3 minutes.
</details>

<details>
<summary><code>calibrate_from_camera(cam_index=0, snapshot_path="calibration_pic.jpg", calib_filename="calibration.json", warmup=10)</code></summary>

1. Captures a snapshot from the chosen camera.  
2. Immediately calls `calibrate_from_file` on that snapshot.  
3. Saves the calibration to JSON.
</details>

<details>
<summary><code>process_image(cam_index: int = 0,
              warmup: int = 10,
              image_path: str | None = None,
              calib_filename: str = "calibration.json") -> np.ndarray</code></summary>

**NEW BEHAVIOR (v2):**

1. **Always captures a fresh snapshot.** If `image_path` is `None`, a timestamped JPEG such as `snapshot_20250419_203015.jpg` is created automatically.  
2. If `calibration.json` is missing, it launches the calibration UI once and stores the result.  
3. Applies the saved calibration rectangle & plate type to the new snapshot and returns a `(3 × N)` matrix of RGB values (rows = R,G,B; columns = wells).

```python
rgb = processor.process_image(cam_index=0, warmup=5)
print(rgb.shape)  # (3, N)
```
</details>

---

### Other Helper Methods

* `resize_to_fit(img, max_width=1280, max_height=720)` → (resized_img, scale)
* `plate_from_trackbar(val)` → "12" | "24" | "48" | "96"
* `get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type)` → List[(x,y)]
* `extract_rgb_values(image, centers, x_offset=0, y_offset=0)` → `np.ndarray` (3 × N)
* `compute_rgb_statistics(rgb_matrix)` → (mean, std, max_dist, min_dist, avg_dist)

---

## End‑to‑End Usage Example

```python
from plate_processor import PlateProcessor

processor = PlateProcessor()

# ‑‑> One‑liner: capture, (auto‑)calibrate, extract colors
rgb = processor.process_image(cam_index=0, warmup=5)
print(rgb.shape)

# Optional statistics
stats = processor.compute_rgb_statistics(rgb)
if stats:
    mean_rgb, std_rgb, max_d, min_d, avg_d = stats
    print(mean_rgb, std_rgb)
```

---

## Workflow Summary

1. **Instantiation** – `processor = PlateProcessor()`.
2. **First call to `process_image()`** – captures a snapshot, opens the calibration UI, stores `calibration.json`, returns RGB data.
3. **Subsequent calls** – captures a new snapshot, silently reuses the existing calibration, and returns updated RGB matrices.

---

## Error Handling & Debugging

- **Snapshot Failure** – raises `FileNotFoundError` if the image cannot be read after capture.
- **Calibration Timeout / Cancel** – if you close the calibration window or let it time out, a `RuntimeError` is raised.
- **Headless Environments** – calibration UI requires a display; use a local machine or remote desktop with GUI support.

---

## Contribution Guidelines

- Keep the drag‑and‑confirm workflow intuitive when modifying the UI.
- Preserve backward compatibility with older `calibration.json` files if you extend the stored metadata.
- Test changes with real hardware; the calibration window must render correctly.
- Update this markdown whenever a public method signature or behavior is changed.

---


# PlateProcessor API Documentation

## Overview

The `PlateProcessor` class provides a high-level interface for **calibrating** and **extracting color data** from plate images. It offers an **interactive calibration window** where you can drag corner points, select a plate type (12, 24, 48, or 96 wells), and save the resulting configuration to a JSON file for subsequent automated processing. It also includes **camera utilities** for taking snapshots and listing available cameras.  

### Key Features

- **Interactive Corner-Dragging UI**  
  Manually calibrate the plate region by dragging four corner points. The code **clamps** these corners to remain inside the image boundaries.  
- **3-Minute Timeout**  
  If you do not confirm within 3 minutes, the UI automatically exits and reports that calibration was canceled.  
- **Automatic JSON Storage**  
  Persists calibration data in a `.json` file, so repeated calibrations are not necessary unless you explicitly choose to re-calibrate.  
- **Well Center Computation**  
  Based on plate type selection, automatically computes the (x, y) center positions of each well.  
- **Color Extraction**  
  Collects RGB values at each well center, optionally averaging a small neighborhood for noise reduction.  
- **Camera Utility**  
  Includes helper functions to list available cameras and capture snapshots via DirectShow.  
- **One-Step Image Processing**  
  The `process_image` method can detect if an image is missing, automatically capture one from the camera, and then request calibration if no valid JSON exists.

---

## Requirements

- **Python 3.x**  
- **OpenCV** (cv2)  
  *Installation:* `pip install opencv-python`
- **NumPy**  
  *Installation:* `pip install numpy`
- **json, os, time** (Built-in Python modules)
- **Optional**: SciPy (if you want to use `compute_rgb_statistics` for advanced distance metrics)  
  *Installation:* `pip install scipy`

---

## Class: PlateProcessor

### Initialization

```python
processor = PlateProcessor()
```

**What It Does:**

- Prepares internal data structures to handle the interactive calibration (corner points, mouse callbacks, confirmation flag).
- Clamps dragged corners within image boundaries.
- Enforces a 3-minute UI timeout if the user does not confirm calibration.

---

## Methods

### Camera & Utility Methods

<details>
<summary><code>list_cameras(max_tested=10) -> List[int]</code></summary>

**Description:**  
Checks camera indices from 0 up to `max_tested-1` using DirectShow. Returns which ones are valid/usable.

**Parameters:**  
- `max_tested`: Maximum number of camera indices to test.

**Returns:**  
- A list of valid camera indices.

**Usage Example:**
```python
valid_cams = PlateProcessor.list_cameras()
print("Available camera indices:", valid_cams)
```
</details>

<details>
<summary><code>take_snapshot(cam_index=0, save_path="snapshot.jpg", warmup_frames=10, properties=None) -> str</code></summary>

**Description:**  
Captures a single image from the specified camera index (using DirectShow on Windows). Supports optional camera property settings (e.g., resolution).

**Parameters:**  
- `cam_index`: Which camera to use (default 0).
- `save_path`: Filename/path to save the snapshot.
- `warmup_frames`: Number of frames to discard before taking the final snapshot (for exposure stabilization).
- `properties`: A dictionary of `cv2.CAP_PROP_*` settings to adjust the camera.

**Returns:**  
- The `save_path` where the snapshot is saved.

**Usage Example:**
```python
snapshot_path = PlateProcessor.take_snapshot(cam_index=1, save_path="my_photo.jpg")
```
</details>

---

### Calibration Methods

<details>
<summary><code>calibrate_from_file(image_path, calib_filename="calibration.json") -> (dict, np.ndarray, float)</code></summary>

**Description:**  
Performs calibration using an existing image on disk. If `calibration.json` exists, the user is prompted to reuse or overwrite it. If it doesn’t exist (or is deleted), an **interactive window** opens, letting you drag corner points and pick a plate type. The code will time out if left unconfirmed for 3 minutes.

**Parameters:**  
- `image_path`: Path to the image file for calibration.
- `calib_filename`: File path for the JSON where calibration data is saved.

**Returns:**  
- A tuple containing:
  1. `calib_data` (dict) with rectangle coordinates and plate type.
  2. `resized_img` (numpy array) used for calibration UI.
  3. `scale` (float) for the resizing factor.

**Usage Example:**
```python
processor = PlateProcessor()
calib_data, resized_img, scale = processor.calibrate_from_file("plate_photo.jpg")
```
</details>

<details>
<summary><code>calibrate_from_camera(cam_index=0, snapshot_path="calibration_pic.jpg", calib_filename="calibration.json", warmup=10)</code></summary>

**Description:**  
1. Captures an image from the specified camera index.  
2. Immediately uses `calibrate_from_file` on that newly captured image.  
3. Saves calibration data to JSON.

**Parameters:**  
- `cam_index`: Camera index for capture.
- `snapshot_path`: Where to save the captured image.
- `calib_filename`: JSON file to store calibration info.
- `warmup`: How many frames to discard before snapping the final photo.

**Usage Example:**  
```python
calib_data, resized, scale = processor.calibrate_from_camera(
    cam_index=0,
    snapshot_path="live_calibration_pic.jpg",
    calib_filename="calibration.json",
    warmup=5
)
```
</details>

<details>
<summary><code>process_image(image_path, calib_filename="calibration.json", cam_index=0, warmup=10) -> np.ndarray</code></summary>

**Description:**  
A one-step method to ensure an image is available (otherwise it snaps one from the camera) and calibration is present (otherwise it launches UI). Then it extracts a `(3, N)` RGB matrix based on the bounding rectangle and plate type from JSON.

**Parameters:**  
- `image_path`: Path to the image or a desired filename for a snapshot.
- `calib_filename`: The JSON file with calibration data.
- `cam_index`: Which camera index to use if we must capture a snapshot.
- `warmup`: Warm-up frames for the camera.

**Returns:**  
- `(3, N)` array of RGB values from the wells.

**Usage Example:**  
```python
rgb_matrix = processor.process_image(
    image_path="my_plate.jpg",
    calib_filename="calibration.json",
    cam_index=0,
    warmup=5
)
print("RGB matrix shape:", rgb_matrix.shape)
```
</details>

---

### Other Helper Methods

<details>
<summary><code>resize_to_fit(img, max_width=1280, max_height=720) -> (np.ndarray, float)</code></summary>

**Description:**  
Resizes an image while preserving aspect ratio so that it does not exceed the given max width and height. Returns the resized image and the scale factor.

</details>

<details>
<summary><code>plate_from_trackbar(val: int) -> str</code></summary>

**Description:**  
Maps the trackbar integer value (0..3) to a string representing the plate type ("12", "24", "48", "96").  
Used internally during calibration UI to reflect the user’s choice.

</details>

<details>
<summary><code>get_well_centers_boxed_grid(x1, y1, x2, y2, plate_type="96") -> List[Tuple[int,int]]</code></summary>

**Description:**  
Computes the (x, y) coordinates of well centers within a rectangular region. The grid size depends on the chosen plate type.

</details>

<details>
<summary><code>extract_rgb_values(image, centers, x_offset=0, y_offset=0) -> np.ndarray</code></summary>

**Description:**  
Extracts the RGB values at each center plus its 4-connected neighbors, then averages them. Returns a `(3, N)` matrix of RGB values.

</details>

<details>
<summary><code>compute_rgb_statistics(rgb_matrix_transposed) -> Tuple</code></summary>

**Description:**  
Given a `(3 x N)` matrix (RGB values), it computes several statistics: mean, standard deviation, and pairwise Euclidean distances (max, min, and average). Requires SciPy.

```python
stats = processor.compute_rgb_statistics(rgb_matrix)
(mean_rgb, std_rgb, max_dist, min_dist, avg_dist) = stats
```
</details>

---

## Usage Example

```python
from plate_processor import PlateProcessor

# 1) Instantiate the class
processor = PlateProcessor()

# 2) Possibly take a snapshot from camera index 0
snapshot_path = processor.take_snapshot(cam_index=0, save_path="snapshot.jpg", warmup_frames=5)

# 3) Calibrate the newly captured image (or an existing image)
calib_data, resized_img, scale = processor.calibrate_from_file(
    image_path="snapshot.jpg",
    calib_filename="calibration.json"
)

# 4) Process the image to extract RGB matrix
rgb_matrix = processor.process_image(
    image_path="snapshot.jpg",
    calib_filename="calibration.json"
)

# 5) Optionally compute statistics
stats = processor.compute_rgb_statistics(rgb_matrix)
if stats is not None:
    mean_rgb, std_rgb, max_dist, min_dist, avg_dist = stats
    print("Mean RGB:", mean_rgb)
    print("Std RGB:", std_rgb)
    print("Max Dist:", max_dist)
    print("Min Dist:", min_dist)
    print("Avg Dist:", avg_dist)
```

### Workflow Summary

1. **Initialization:**  
   Create an instance of `PlateProcessor`. This sets up the corner-dragging UI logic, camera utilities, and a 3-minute calibration timeout.

2. **Image Capture (Optional):**  
   Use `take_snapshot()` to capture a photo from a chosen camera index.

3. **Calibration:**  
   - Use `calibrate_from_file(...)` to open a window if needed.  
   - Drag corners, choose plate type, confirm within 3 minutes or it times out.  
   - The bounding rectangle + plate type are stored in a JSON file.

4. **Color Extraction:**  
   - `process_image()` loads calibration from JSON. If the image doesn’t exist, it captures one automatically.  
   - Extracts `(3, N)` RGB data from each well center.

5. **Statistics (Optional):**  
   - `compute_rgb_statistics()` can produce mean/std of the colors, plus pairwise distances for advanced analysis.

---

## Error Handling & Debugging

- **Image Load Failures**  
  If `cv2.imread()` fails (e.g., invalid path), the code raises a `FileNotFoundError`.  
- **Calibration Timeout**  
  If the user does not confirm within 3 minutes during calibration, it auto-cancels and returns `None`.  
- **Headless Environments**  
  Running in a context with no GUI (e.g., a remote server) will prevent windows from appearing. Use a local environment with display capability.  

---

## Contribution Guidelines

- **Modifying the Calibration UI**  
  If you add features or tweak the corner logic, ensure the “drag + confirm” user experience remains intuitive.  
- **Extending Calibration Data**  
  If adding new metadata to `calibration.json`, maintain backward compatibility so older JSON files remain valid.  
- **Testing & Validation**  
  Always test in an environment where OpenCV windows can appear. Confirm your changes with real images and hardware.  
- **Documentation**  
  Update this file if any public methods change (signatures, new parameters, etc.). Provide usage examples for new features.

---

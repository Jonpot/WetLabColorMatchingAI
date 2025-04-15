# PlateProcessor API Documentation

## Overview

The `PlateProcessor` class provides a high-level interface for calibrating and extracting color data from plate images. It offers an interactive calibration window where you can drag corner points, select a plate type (12, 24, 48, or 96 wells), and save the resulting configuration to a JSON file for subsequent automated processing. It also includes camera utilities for taking snapshots and listing available cameras.

### Key Features

- **Calibration with Interactive UI**  
  Allows you to manually calibrate the plate region in an image by dragging four corner points.  
- **Automatic JSON Storage**  
  Persists calibration data in a `.json` file, so repeated calibrations are not necessary unless you explicitly choose to re-calibrate.  
- **Well Center Computation**  
  Based on plate type selection, automatically computes the (x, y) center positions of each well.  
- **Color Extraction**  
  Collects RGB values at each well center, optionally averaging a small neighborhood for noise reduction.  
- **Camera Utility**  
  Includes helper functions to list available cameras and capture snapshots via DirectShow.  

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

PlateProcessor()

**What It Does:**

- Prepares internal data structures to handle the interactive calibration (corner points, mouse callbacks, confirmation flag).
- Offers a unified interface for camera operations, color extraction, and calibration-based ROI processing.

---

## Methods

### Camera & Utility Methods

<details>
<summary><code>list_cameras(max_tested=10) -> List[int]</code></summary>

**Description:**  
Checks camera indices from 0 up to `max_tested-1` to see which ones are valid (usable) on the system.

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
<summary><code>calibrate(image_path, calib_filename="calibration.json") -> (dict, np.ndarray, float)</code></summary>

**Description:**  
Ensures that calibration data (bounding rectangle + plate type) is available for the given image. If `calibration.json` exists, it prompts you to use the old data or overwrite it. If it doesn’t exist, an interactive UI launches to let you drag the four corners and select a plate type.

**Parameters:**  
- `image_path`: Path to the image file to display for calibration.
- `calib_filename`: File path for the JSON where calibration data is saved.

**Returns:**  
- A tuple containing:
  1. `calib_data` (dict): The JSON calibration data (rectangle coords, plate type).
  2. `resized_img` (numpy array): The image resized for the calibration window.
  3. `scale` (float): The scale factor used to resize the original image.

**Usage Example:**
```python
processor = PlateProcessor()
calib_data, resized_img, scale = processor.calibrate("pictures/4.jpg", "calibration.json")
```
</details>

<details>
<summary><code>process_image(image_path, calib_filename="calibration.json") -> np.ndarray</code></summary>

**Description:**  
1. Checks if calibration data exists.  
2. Loads the calibrated rectangle & plate type.  
3. Computes well centers.  
4. Extracts the (3 x N) RGB matrix from the calibrated region of the resized image.  

**Parameters:**  
- `image_path`: Path to the target image.
- `calib_filename`: The JSON file with calibration data.

**Returns:**  
- A NumPy array with shape `(3, N)` representing the RGB values for each well.

**Usage Example:**
```python
processor = PlateProcessor()
rgb_matrix = processor.process_image("pictures/4.jpg", "calibration.json")
print("RGB matrix shape:", rgb_matrix.shape)
```
</details>

---

### Other Helper Methods

<details>
<summary><code>resize_to_fit(img, max_width=1280, max_height=720) -> (np.ndarray, float)</code></summary>

**Description:**  
Resizes an image while preserving aspect ratio so that it does not exceed the given max width and height. Returns the resized image and the scale factor.

**Usage Example:**
```python
resized_img, scale = PlateProcessor.resize_to_fit(img, 1280, 720)
```
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

# 2) (Optional) Take a snapshot from camera index 0
snapshot_path = processor.take_snapshot(cam_index=0, save_path="snapshot.jpg", warmup_frames=5)

# 3) Calibrate the newly captured image (or any chosen image)
calib_data, resized_img, scale = processor.calibrate("snapshot.jpg", "calibration.json")

# 4) Process the image using the saved calibration to extract RGB matrix
rgb_matrix = processor.process_image("snapshot.jpg", "calibration.json")

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
   Create an instance of `PlateProcessor`. This prepares the interactive calibration UI framework and camera helpers.

2. **Image Acquisition (Optional):**  
   Use `take_snapshot()` to capture a photo from a connected camera.

3. **Calibration (Only if needed):**  
   - Run `calibrate()`. If a `.json` calibration file does not exist, an interactive window opens.  
   - Drag the corner points around your plate, choose the plate type, and click “Confirm.”  
   - The calibration data is then stored in the specified JSON file.

4. **Color Extraction:**  
   - Run `process_image()` to load the calibration data, compute the well centers, and extract RGB values.  
   - The result is a `(3, N)` NumPy array representing the colors of each well.

5. **Analysis (Optional):**  
   - Call `compute_rgb_statistics()` to get summary statistics (mean, std, distance metrics).  

---

## Error Handling & Debugging

- **Image Load Failures**  
  If `cv2.imread()` fails (e.g., invalid path), the code raises a `FileNotFoundError`.  
- **Calibration Window**  
  Make sure you provide a valid image for calibration.  
- **UI Events**  
  In case the calibration window doesn’t appear, ensure you’re not running in a headless environment.

---

## Contribution Guidelines

- **Enhancing Calibration UI**  
  If you add new features or want to modify the corner-point logic, keep the draggable system and button controls consistent so the user experience remains predictable.

- **Maintaining JSON Structure**  
  If you add more data to `calibration.json` (e.g., additional metadata), ensure backward compatibility, so old JSON files still load properly.

- **Testing**  
  Always test in an environment where you can display OpenCV windows. Validate new methods with real images.

- **Documentation**  
  Update this `.md` whenever you add or remove functionalities, and provide usage examples for new methods.

---

*This documentation outlines the core PlateProcessor API, explaining each method, its parameters, and usage examples. For more advanced or custom workflows, you can extend `PlateProcessor` or integrate it with higher-level automation scripts in your pipeline.*  

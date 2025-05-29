# WetLabColorMatchingAI

WetLabColorMatchingAI automates color mixing experiments on an Opentrons OT-2 robot using a webcam for feedback. The project aims to learn dye recipes that reproduce arbitrary target colors.  A recent cleanup reorganized the repository so each component now lives in its own folder, which should make the workflow clearer for new contributors.

## Repository structure

- **app.py** – Streamlit interface for manually mixing colors and competing
  against the OT-2.
- **main_active_learning.py** – Provides reusable functions to run the active
  learning loop and can also be executed as a script for a full plate run.
- **active_learning/** – Optimizers and algorithms that decide which dye volumes
  to test next.
- **camera/** – Camera calibration utilities and example data used to measure
  plate colors.  The :class:`PlateProcessor` supports a ``virtual_mode`` that
  skips the webcam and returns an all white plate for testing.
- **robot/** – OT-2 helper functions and argument files shared by the different
  protocols.
- **remote/** – Protocols intended to be executed directly on the OT-2.
- **color_space_research/** – Notebooks and scripts for exploring color spaces
  and analyzing lighting conditions.
- **utils/** – Local test scripts and helper tools.
- **old_tests/** – Legacy experiments that are kept for reference.
- **secret/** – SSH keys used to connect to the OT-2 (not included in the repo).

## Installation

All scripts require Python 3. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Connecting to the OT-2

Both the Streamlit app and the learning pipeline connect to the OT‑2 over SSH
using `paramiko`. Edit the hostname, username and SSH key path in the scripts if
your robot uses different settings. Make sure your workstation can reach the
robot on the network and that the SSH key is stored in the `secret/` folder.

## Usage

### Streamlit interface

```bash
streamlit run app.py
```

This launches the web UI where you can enter dye volumes and see the measured
colors.

### Active learning

```bash
python main_active_learning.py
```

The script iterates through plate rows, mixing and measuring until it matches
each target color or the iteration limit is reached.

Both ``active_learn_row`` and ``run_active_learning`` can also be imported from
``main_active_learning`` if you wish to integrate the optimisation loop into
another application.

# WetLabColorMatchingAI

This repository contains experiments for mixing liquid dyes with an Opentrons OT-2 robot and a webcam.
The goal is to automatically create recipes that reproduce a target color.

## Key scripts

- **app.py** – Streamlit interface that allows a user to compete against the OT‑2 in a color mixing challenge.
- **main_active_learning.py** – Runs an active learning pipeline that autonomously chooses dye volumes, instructs the OT‑2, photographs the plate and logs results.

## Installation

The scripts require Python 3. Install the main dependencies with:

```bash
pip install paramiko scp opencv-python streamlit scikit-learn
```

Additional packages such as `numpy`, `matplotlib` and others that ship with Python scientific stacks may also be required.

## Connecting to the OT‑2

Both the Streamlit app and the learning pipeline connect to the OT‑2 over SSH using `paramiko`.
Edit the `hostname`, `username` and SSH key path in the scripts if your robot uses different settings.
Make sure your workstation can reach the OT‑2 on the network and that the SSH key is stored in the `secret/` folder.

## Usage

### Streamlit interface

```bash
streamlit run app.py
```

This launches the web UI where you can enter dye volumes and see the measured colors.

### Active learning

```bash
python main_active_learning.py
```

The script will iterate through plate rows, mixing and measuring until it matches each target color or the iteration limit is reached.



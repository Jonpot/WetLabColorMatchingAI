from active_learning.color_learning import ColorLearningOptimizer
import numpy as np
import time
import random

# --- Mock OT2Manager class ---
class MockOT2Manager:
    def __init__(self, hostname=None, username=None, key_filename=None, password=None):
        print(f"[MOCK] Connecting to OT2 robot at {hostname}...")
        self.actions = []
    
    def add_add_color_action(self, color_slot, plate_well, volume):
        print(f"[MOCK] Adding action: Add {volume}μL dye from slot {color_slot} to well {plate_well}")
        self.actions.append({"type": "add_color", "slot": color_slot, "well": plate_well, "volume": volume})
    
    def execute_actions_on_remote(self):
        print(f"[MOCK] Executing {len(self.actions)} actions...")
        time.sleep(0.5)  # Simulate execution time
        self.actions = []
    
    def add_close_action(self):
        print("[MOCK] Adding close action")

# --- Mock PlateProcessor class ---
class MockPlateProcessor:
    def __init__(self):
        # Create some random RGB values for target colors and initial colors
        self.target_colors = [np.random.randint(0, 256, 3) for _ in range(7)]  # 7 target colors
        self.current_colors = []
        for i in range(7):
            row_colors = [self.target_colors[i]]  # First column is the target color
            for j in range(11):  # Add 11 empty columns for each row (to be filled later)
                row_colors.append(np.array([255, 255, 255]))  # Initially white
            self.current_colors.append(row_colors)
        print("[MOCK] Initialized simulated plate processor")
    
    def process_image(self, cam_index=0, warmup=None):
        print(f"[MOCK] Processing image, camera index = {cam_index}")
        time.sleep(0.5)  # Simulate image processing time
        return self.current_colors
    
    def update_well_color(self, row_idx, col_idx, dye_volumes):
        """Simulate color change based on added dye volumes"""
        # Simple simulation: Start from white and move toward target color based on dye volumes
        total_volume = sum(dye_volumes)
        if total_volume > 0:
            # Calculate mixing ratio with target color
            target = self.target_colors[row_idx]
            # Calculate degree of approach to target color (add randomness to simulate imprecise mixing)
            approach_factor = min(sum(dye_volumes) / 100, 1.0) * (0.7 + 0.3 * random.random())
            # Simulate color change (move from white toward target color)
            white = np.array([255, 255, 255])
            new_color = white * (1 - approach_factor) + target * approach_factor
            # Add some random variation to simulate experimental error
            noise = np.random.normal(0, 10, 3)
            new_color = np.clip(new_color + noise, 0, 255).astype(int)
            self.current_colors[row_idx][col_idx] = new_color

# --- Initialize mock connection ---
robot = MockOT2Manager(
    hostname="169.254.122.0",  # Replace with your OT-2's IP address
    username="root",
    key_filename="secret/ot2_ssh_key",  
    password="lemos" 
)

# Define dye reservoir slots and plate rows
color_slots = ['7', '8', '9']
plate_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
MAX_WELL_VOLUME = 200
TOLERANCE = 30
MIN_STEP = 1
MAX_ITERATIONS = 11

# Initialize optimizer
optimizer = ColorLearningOptimizer(
    dye_count=len(color_slots),
    max_well_volume=MAX_WELL_VOLUME,
    step=MIN_STEP,
    tolerance=TOLERANCE,
    min_required_volume=20,
    optimization_mode="unmixing"  # or "random_forest"
)

# --- Start Active Learning Loop ---

processor = MockPlateProcessor()
color_data = processor.process_image(cam_index=0, warmup=5)  # Read plate color

for row in plate_rows:
    print(f"\nProcessing row {row}")

    # Reset optimizer for new row
    optimizer.reset()

    row_idx = ord(row) - ord('A')
    target_color = color_data[row_idx][0]  # Column 1 is the target color
    print(f"Target color RGB: {target_color}")

    completed = False
    current_iteration = 0

    while not completed and current_iteration < MAX_ITERATIONS:
        column = current_iteration + 2  # Start from column 2
        well_coordinate = f"{row}{column}"

        print(f"\nRow {row} Iteration {current_iteration + 1}: using well {well_coordinate}")

        # Suggest next volumes to add
        volumes = optimizer.suggest_next_experiment(target_color)
        print(f"Suggested dye volumes: {volumes}")

        # Queue add_color actions
        for i, volume in enumerate(volumes):
            if volume > 0:
                robot.add_add_color_action(
                    color_slot=color_slots[i],
                    plate_well=well_coordinate,
                    volume=volume
                )

        # Send actions and execute
        robot.execute_actions_on_remote()

        # Simulate color change
        processor.update_well_color(row_idx, column - 1, volumes)

        # Wait for dye to stabilize
        time.sleep(0.5)

        # Read updated plate color
        color_data = processor.process_image(cam_index=0)
        measured_color = color_data[row_idx][column - 1]

        print(f"Measured color RGB: {measured_color}")

        # Add experimental data to optimizer
        optimizer.add_data(volumes, measured_color)

        # Calculate distance
        distance = optimizer.calculate_distance(measured_color, target_color)
        print(f"Distance to target: {distance:.2f}")

        # Check if matched
        if optimizer.within_tolerance(measured_color, target_color):
            print(f"✓ Target matched for row {row}! Final recipe: {volumes}")
            completed = True
        else:
            current_iteration += 1

print("\nColor Matching Complete!")

# Close session
robot.add_close_action()
robot.execute_actions_on_remote() 
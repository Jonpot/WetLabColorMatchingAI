from ot2_utils import OT2Manager
from color_learning import ColorLearningOptimizer
from camera_w_calibration import PlateProcessor
import time

# --- Initialize Connection ---
robot = OT2Manager(
    hostname="169.254.122.0", 
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

processor = PlateProcessor()
color_data = processor.process_image(cam_index=0, warmup=5)# Read plate color

for row in plate_rows:
    print(f"Processing row {row}")

    # Reset optimizer for new row
    optimizer.reset()

    row_idx = ord(row) - ord('A')
    target_color = color_data[row_idx][0]  # Column 1 is the target color

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

        # Wait for dye to stabilize
        time.sleep(3)

        # Read updated plate color
        color_data = processor.process_image(cam_index=0)
        measured_color = color_data[row_idx][column - 1]

        print(f"Measured color: {measured_color}")

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

print("Color Matching Complete!")

# Close session
robot.add_close_action()
robot.execute_actions_on_remote()

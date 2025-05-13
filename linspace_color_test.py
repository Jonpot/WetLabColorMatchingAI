from ot2_utils import OT2Manager
import numpy as np
import random
import pandas as pd
from camera_w_calibration import PlateProcessor
CAM_INDEX = 2  # camera index for the plate processor

# Define plate rows and columns
plate_rows_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
plate_col_letters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

# Define color slots
color_slots = {"red": "7", "green": "8", "blue": "9", "water": "11"}

# Instantiate OT2Manager and PlateProcessor
try:
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None, reduced_tips_info=len(color_slots))
except Exception as e:
    print(f"Error initializing OT2Manager: {e}")
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None, reduced_tips_info=len(color_slots))

processor = PlateProcessor()

# Turn on lights
robot.add_turn_on_lights_action()

# Create a 96 well plate
def create_linspace_recipe(start: int, end: int, num: int, color: str, total_volume: int = 200) -> list[dict[str, int]]:
    """Create a list of linear space recipes for the given start and end concentrations.
    Every recipe will have a total volume of 200uL, supplemented with water."""
    linspace = np.linspace(start, end, num=num)
    recipes = []
    for concentration in linspace:
        recipe = {color: int(concentration), "water": total_volume - int(concentration)}
        recipes.append(recipe)
    return recipes

def create_replicate_linspace_recipe(start: int, end: int, num: int, color: str, total_volume: int = 200, replicates: int = 4) -> list[dict[str, int]]:
    """Create a list of linear space recipe replicates for the given start and end concentrations.
    Every recipe will have a total volume of 200uL, supplemented with water."""
    recipes = []
    for _ in range(replicates):
        recipes.extend(create_linspace_recipe(start, end, num, color, total_volume))
    return recipes

def create_random_recipe(colors: list[str], total_volume: int = 200, min_volume: int = 20) -> dict[str, int]:
    """Create a random recipe for the given colors with a total volume of 200uL.
    
    Note that while the default minimum volume is "20uL", that's only the minimum _if the color is present_.
    If the color is not present, it will be 0uL.
    """
    recipe = {}
    total_volume_remaining = total_volume
    for color in colors:
        if random.random() < 0.5:  # 50% chance to include the color
            volume = random.randint(min_volume, total_volume_remaining)
            recipe[color] = volume
            total_volume_remaining -= volume
        else:
            recipe[color] = 0
    recipe["water"] = total_volume_remaining
    return recipe

# Create recipes:
# Quadruplicate linear space recipes with 4 wells each for each primary color
# With 3 primary colors, this will 4*4*3 = 48 wells
# Then, with the remaining 48 wells, create random recipes 
protocol_recipes = []
for color in ["red", "green", "blue"]:
    protocol_recipes.extend(create_replicate_linspace_recipe(0, 200, 4, color, total_volume=200, replicates=4))
# Add random recipes for the remaining wells
for _ in range(48):
    protocol_recipes.append(create_random_recipe(["red", "green", "blue"], total_volume=200, min_volume=20))

# Now, randomly shuffle the recipes then add them to the plate
random.shuffle(protocol_recipes)
# Add recipes to the plate as well as a dataframe
# df will include the following columns:
# well, red_vol, green_vol, blue_vol, water_vol, measured_red, measured_green, measured_blue
# The measured columns will be filled in later
df = pd.DataFrame(columns=["well", "red_vol", "green_vol", "blue_vol", "water_vol", "measured_red", "measured_green", "measured_blue"])

for i, recipe in enumerate(protocol_recipes):
    row = plate_rows_letters[i // 12]
    col = plate_col_letters[i % 12]
    well = f"{row}{col}"
    for color, volume in recipe.items():
        if color in color_slots:
            robot.add_add_color_action(
                color = color_slots[color],
                plate_well=well,
                volume = volume,
            )
    robot.add_mix_action(
        plate_well=well,
        volume=100,
        repetitions=3,
    )

    df.loc[i] = [
        well,
        recipe.get("red", 0),
        recipe.get("green", 0),
        recipe.get("blue", 0),
        recipe.get("water", 0),
        None,  # Placeholder for measured red
        None,  # Placeholder for measured green
        None,  # Placeholder for measured blue
    ]


# Execute actions
robot.execute_actions_on_remote()

# Now, measure the colors in each well
measured_plate = processor.process_image(cam_index=CAM_INDEX)
for i, recipe in enumerate(protocol_recipes):
    row = plate_rows_letters[i // 12]
    col = plate_col_letters[i % 12]
    well = f"{row}{col}"
    row_idx = ord(row) - ord("A")
    col_idx = int(col) - 1
    rgb = measured_plate[row_idx, col_idx]
    
    # Store the measured RGB values in the dataframe
    df.at[i, "measured_red"] = rgb[0]
    df.at[i, "measured_green"] = rgb[1]
    df.at[i, "measured_blue"] = rgb[2]


# Turn off lights and close connection
robot.add_turn_off_lights_action()
robot.add_close_action()
robot.execute_actions_on_remote()


# Save the dataframe to a CSV file
df.to_csv("linspace_data.csv", index=False)

# Now, analyze the data and plot the results
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

# Set the color palette
palette = sns.color_palette("husl", 3)
# Create a color map for the colors
cmap = mcolors.ListedColormap(palette)
# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
# Create a scatter plot for the measured colors
for i, color in enumerate(["red", "green", "blue"]):
    ax.scatter(df[f"measured_{color}"], df[f"measured_{color}"], color=palette[i], label=color, alpha=0.5)
# Set the axis labels and title
ax.set_xlabel("Measured Red")
ax.set_ylabel("Measured Green")
ax.set_title("Measured Colors")
# Set the x and y axis limits
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
# Set the x and y axis ticks
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
# Add a grid
ax.grid(True, linestyle='--', alpha=0.5)
# Add a legend
ax.legend()
# Show the plot
plt.show()
# Save the plot
plt.savefig("measured_colors.png", dpi=300)

# Create a heatmap of the measured colors
fig, ax = plt.subplots(figsize=(12, 8))
# Create a heatmap for the measured colors
for x_color, y_color in zip(["red", "green", "blue"], ["green", "blue", "red"]):
    heatmap_data = df.pivot("well", f"measured_{x_color}", f"measured_{y_color}")
    sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt=".1f", cbar_kws={"label": f"Measured Color Value between {x_color} and {y_color}"}, ax=ax, linewidths=0.5)
    # Set the axis labels and title
    ax.set_xlabel(f"Measured {x_color}")
    ax.set_ylabel(f"Measured {y_color}")
    ax.set_title(f"Measured Colors Heatmap: {x_color} vs {y_color}")
    # Set the x and y axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.5)
    # Show the plot
    plt.show()
    # Save the plot
    plt.savefig(f"measured_colors_heatmap_{x_color}_{y_color}.png", dpi=300)
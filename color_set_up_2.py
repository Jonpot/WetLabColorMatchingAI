from ot2_utils import OT2Manager
import numpy as np
import random

try:
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None)
except Exception as e:
    print(f"Error initializing OT2Manager: {e}")
    robot = OT2Manager(hostname="172.26.192.201", username="root", key_filename="secret/ot2_ssh_key_remote", password=None)

# Define plate rows and columns
plate_rows_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
plate_col_letters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

# Define color slots
color_slots = {"red": "7", "green": "8", "blue": "9", "water": "11"}

# Turn on lights
robot.add_turn_on_lights_action()

# Part 1: Add specific recipes for rows A-D
# Red in columns 1-4
red_recipes = [
    {"red": 20, "water": 180},  # A1
    {"red": 80, "water": 120},  # A2
    {"red": 120, "water": 80},  # A3
    {"red": 200, "water": 0},   # A4
]

for row in ["A", "B", "C", "D"]:
    for col_idx, recipe in enumerate(red_recipes, start=1):
        well = f"{row}{col_idx}"
        for color, volume in recipe.items():
            if volume > 0:
                robot.add_add_color_action(color_slot=color_slots[color], plate_well=well, volume=volume)

# Green in columns 5-8
green_recipes = red_recipes  # Same recipes as red
for row in ["A", "B", "C", "D"]:
    for col_idx, recipe in enumerate(green_recipes, start=5):
        well = f"{row}{col_idx}"
        for color, volume in recipe.items():
            if volume > 0:
                robot.add_add_color_action(color_slot=color_slots[color], plate_well=well, volume=volume)

# Blue in columns 9-12
blue_recipes = red_recipes  # Same recipes as red
for row in ["A", "B", "C", "D"]:
    for col_idx, recipe in enumerate(blue_recipes, start=9):
        well = f"{row}{col_idx}"
        for color, volume in recipe.items():
            if volume > 0:
                robot.add_add_color_action(color_slot=color_slots[color], plate_well=well, volume=volume)

# Part 2: Randomize recipes for rows E-H
randomized_recipes = {}
for row in ["E", "F", "G", "H"]:
    for col in plate_col_letters:
        well = f"{row}{col}"
        recipe = {}
        total_volume = 0
        while total_volume < 200:  # Ensure total volume is 200
            for color in ["red", "green", "blue", "water"]:
                volume = random.uniform(0, 200 - total_volume)
                if 0 < volume < 10:
                    volume = 0
                elif 10 <= volume < 20:
                    volume = 20
                recipe[color] = recipe.get(color, 0) + volume
                total_volume += volume
                if total_volume >= 200:
                    break
        randomized_recipes[well] = recipe
        for color, volume in recipe.items():
            if volume > 0:
                robot.add_add_color_action(color_slot=color_slots[color], plate_well=well, volume=volume)

# Part 3: Generate a list of wells and their corresponding recipes
with open("well_recipes.txt", "w") as f:
    for well, recipe in randomized_recipes.items():
        recipe_str = ", ".join([f"{color}: {volume}uL" for color, volume in recipe.items() if volume > 0])
        f.write(f"{well}: {recipe_str}\n")
        print(f"{well}: {recipe_str}")

# Execute actions
robot.execute_actions_on_remote()

# Turn off lights and close connection
robot.add_turn_off_lights_action()
robot.add_close_action()
robot.execute_actions_on_remote()
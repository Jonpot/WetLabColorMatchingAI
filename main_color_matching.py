import json
from pathlib import Path
import sys
from opentrons import protocol_api

color_slots = ['7','8','9']
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

metadata = {
    'protocolName': 'Color Matching v0.3',
    'author': 'CMU Automated Lab',
    'description': 'Attempts to robotically match colors using active learning.',
}

requirements = {'robotType': 'OT-2', 'apiLevel': '2.19'}


def run(protocol: protocol_api.ProtocolContext) -> None:
    """Defines the testing protocol."""

    class Well:
        """
        Represents a well on a plate.
        """
        def __init__(self, max_volume: float, volume: float = 0):
            self.max_volume = max_volume
            self.volume = volume

    class Plate:
        """
        Represents a plate with wells.
        """
        def __init__(self, labware: protocol_api.Labware, rows: int, columns: int, well_max_volume: float):
            self.labware = labware
            self.rows = rows
            self.columns = columns
            self.wells = {f"{ascii_uppercase[row]}{column + 1}": Well(well_max_volume) for row in range(rows) for column in range(columns)}

        def get_well(self, row: int, column: int) -> Well:
            return self.wells[row][column]

    def get_filename() -> str:
        filename = "color_matching_tiprack.json'"
        # use Path.home() on Mac, Linux, and on the robot   
        output_file_destination_path = Path.home().joinpath(
            filename
        )

        # on windows put the file into a directory
        # we have high confidence the app can write a file to.
        if sys.platform.startswith("win"):
            output_file_destination_path = Path.home().joinpath(
                "AppData",
                "Roaming",
                "Opentrons",
                filename,
            )
        # print in the run log where the output file is
        protocol.comment(f"output file path = {output_file_destination_path}")
        return output_file_destination_path

    def setup(plate_type: str = "corning_96_wellplate_360ul_flat") -> tuple[dict[str, protocol_api.Labware],
                                                                      Plate, protocol_api.InstrumentContext,
                                                                      list[bool],
                                                                      list[protocol_api.Labware]]:
        """
        Loads labware and instruments for the protocol.

        :param plate_type: The type of plate to use, as per the Opentrons API.
        """
        tipracks: list[protocol_api.Labware] = [protocol.load_labware('opentrons_96_tiprack_300ul', location='3')]

        # Some tips may be missing, so we need to update the current state of the tip rack from
        # the file. This is necessary to avoid the robot trying to use tips that are not present.

        # Check ./color_matching_tiprack.json exists, if not make it and assume full rack
        try:
            with open(get_filename(), 'r') as f:
                tiprack_state = json.load(f)
        except FileNotFoundError:
            tiprack_state = [True] * 96
        except json.JSONDecodeError:
            tiprack_state = [True] * 96

        colors: dict[str, protocol_api.Labware] = {}
        for slot in color_slots:
            colors[slot] = protocol.load_labware('nest_1_reservoir_290ml', location=str(slot))['A1']

        plate_labware = protocol.load_labware(plate_type, label="Dye Plate", location='1')
        plate = Plate(plate_labware, len(plate_labware.rows()), len(plate_labware.columns()), plate_labware.wells()[0].max_volume)

        pipette = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=tipracks)

        off_deck_tipracks = []
        for _ in range(10): # arbitrarily high number of tip boxes
            # these tip boxes will be replaced as needed
            off_deck_tipracks.append(protocol.load_labware('opentrons_96_tiprack_300ul', location=protocol_api.OFF_DECK))

        return colors, plate, pipette, tiprack_state, off_deck_tipracks
    
    def get_plate_type() -> str:
        """
        Uses the attached camera to determine the type of plate being used.
        """
        raise NotImplementedError

    #plate_type = get_plate_type()
    plate_type = "corning_96_wellplate_360ul_flat" # TODO: Remove this line when get_plate_type is implemented.
    colors, plate, pipette, tiprack_state, off_deck_tipracks = setup(plate_type)

    def get_color() -> list[list[list[float]]]:
        """
        Uses the attached camera to determine the color of the dye in each well on the plate.

        :return: An array of RGB values for each well on the plate.
        """
        raise NotImplementedError

    def pick_up_tip(tiprack_state: list[bool]) -> list[bool]:
        """
        Picks up a tip from the tip rack.
        """
        try:
            next_well = tiprack_state.index(True)
        except ValueError:
            protocol.comment("No tips left in the tip rack, switching to new rack.")
            on_deck_position = pipette.tip_racks[0].parent
            new_tiprack = off_deck_tipracks.pop()
            protocol.move_labware(labware=pipette.tip_racks[0], new_location=protocol_api.OFF_DECK)
            protocol.move_labware(labware=new_tiprack, new_location=on_deck_position)
            pipette.tip_racks[0] = new_tiprack
            next_well = 0
            tiprack_state = [True] * 96
        pipette.pick_up_tip(location=pipette.tip_racks[0].well(next_well))
        tiprack_state[next_well] = False

        return tiprack_state

    def return_tip() -> None:
        """
        Returns the tip to the tip rack.
        """
        pipette.drop_tip()

    def add_color(
            color_slot: str | int,
            plate_well: str,
            volume: float,
            tiprack_state: list[bool]) -> list[bool]:
        """
        Adds a color to the plate at the specified well.

        :param color_slot: The slot of the color reservoir.
        :param plate_well: The well of the plate to add the color to.
        :param volume: The volume of the color to add.

        :raises ValueError: If the well is already full.
        """
        if volume + plate.wells[plate_well].volume > plate.wells[plate_well].max_volume:
            raise ValueError("Cannot add color to well; well is full.")

        tiprack_state = pick_up_tip(tiprack_state)
        pipette.aspirate(volume, colors[color_slot])
        pipette.touch_tip(plate.labware[plate_well], v_offset=95, radius=0) # necessary to avoid crashing against the large adapter
        pipette.dispense(volume, plate.labware[plate_well].bottom(z=81))

        plate.wells[plate_well].volume += volume

        # Quick mix (has to be manual because the default mix function doesn't work with the large adapter)
        pipette.aspirate(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.dispense(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.aspirate(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.dispense(volume/2, plate.labware[plate_well].bottom(z=81))

        return_tip()

        return tiprack_state

    def close(tiprack_state: list[bool]) -> None:
        """
        Closes the protocol, saving the state of the tip rack.
        """
        if  protocol.is_simulating():
            # don't save tiprack state in simulation
            return
        with open(get_filename(), 'w') as f:
            json.dump(tiprack_state, fp=f)

    # TESTING
    tiprack_state = add_color('7', 'A1', 100, tiprack_state)
    tiprack_state = add_color('8', 'A1', 200, tiprack_state)
    tiprack_state = add_color('9', 'A2', 300, tiprack_state)

    close(tiprack_state)
    # return 

    def active_learning() -> bool:
        """
        The main loop of the protocol, which uses active learning to determine the best color match.
        Each row's first position (column 1) contains the target color.
        Columns 2-12 are used for experiment iterations.

        Returns:
            bool: True if all rows are completed, False otherwise.
        """
        #TODO: Change according to the plate type
        MAX_WELL_VOLUME = 200 
        TOLERANCE = 30          
        MIN_STEP = 1        
        MAX_ITERATIONS = 11   

        if not hasattr(active_learning, "initialized"):
            active_learning.initialized = True
            active_learning.current_row = 'A'
            active_learning.rows_to_process = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] #TODO: Change according to the plate type
            active_learning.row_data = {}

            color_data = get_color()
            print("Reading target colors from column 1...")

            for row in active_learning.rows_to_process:
                row_idx = ord(row) - ord('A')
                target_color = color_data[row_idx][0]
                print(f"Row {row} target color: {target_color}")

                covering_combos = generate_diverse_covering_combinations(
                    dye_count=len(color_slots),
                    n_combinations=4,
                    max_volume=MAX_WELL_VOLUME,
                    step=MIN_STEP
                )

                active_learning.row_data[row] = {
                    'target_color': target_color,
                    'current_iteration': 0,
                    'X_train': [],
                    'Y_train': [],
                    'best_match': None,
                    'best_distance': float('inf'),
                    'completed': False,
                    'covering_combinations': covering_combos
                }

        row = active_learning.current_row
        row_data = active_learning.row_data[row]

        while row_data['completed'] and row in active_learning.rows_to_process:
            current_idx = active_learning.rows_to_process.index(row)
            if current_idx + 1 < len(active_learning.rows_to_process):
                active_learning.current_row = active_learning.rows_to_process[current_idx + 1]
                row = active_learning.current_row
                row_data = active_learning.row_data[row]
            else:
                print("All rows completed!")
                return True

        if row_data['current_iteration'] >= MAX_ITERATIONS:
            print(f"Row {row} reached max iterations. Best match: {row_data['best_match']}")
            row_data['completed'] = True
            current_idx = active_learning.rows_to_process.index(row)
            if current_idx + 1 < len(active_learning.rows_to_process):
                active_learning.current_row = active_learning.rows_to_process[current_idx + 1]
            else:
                print("All rows completed!")
                return True
            return False

        column = row_data['current_iteration'] + 2
        well_coordinate = f"{row}{column}"
        print(f"Row {row} - Iteration {row_data['current_iteration']+1} - Using well {well_coordinate}")

        if row_data['current_iteration'] < len(row_data['covering_combinations']):
            volumes = row_data['covering_combinations'][row_data['current_iteration']]
        else:
            volumes = random_forest_optimize_next_experiment(
                row_data['X_train'],
                row_data['Y_train'],
                row_data['target_color'],
                len(color_slots),
                MAX_WELL_VOLUME,
                MIN_STEP,
                MAX_ITERATIONS
            )


        print(f"Adding dye combination: {volumes}")
        for i, volume in enumerate(volumes):
            if volume > 0:
                add_color(color_slots[i], well_coordinate, volume)

        color_data = get_color()
        row_idx = ord(row) - ord('A')
        col_idx = column - 1
        measured_color = color_data[row_idx][col_idx]

        print(f"Measured color: {measured_color}")

        target_color = row_data['target_color']
        distance = calculate_distance_to_target(measured_color, target_color)
        print(f"Distance to target: {distance:.2f}")

        row_data['X_train'].append(volumes)
        row_data['Y_train'].append(measured_color)

        if distance < row_data['best_distance']:
            row_data['best_distance'] = distance
            row_data['best_match'] = volumes
            print(f"New best match! Distance: {distance:.2f}")

        if within_tolerance(measured_color, target_color, TOLERANCE):
            print(f"✓ Target matched for row {row}! Recipe: {volumes}")
            row_data['completed'] = True
            current_idx = active_learning.rows_to_process.index(row)
            if current_idx + 1 < len(active_learning.rows_to_process):
                active_learning.current_row = active_learning.rows_to_process[current_idx + 1]
            else:
                print("All rows completed!")
                return True
        else:
            row_data['current_iteration'] += 1

        return False


    # Main Loop
    i = 0
    while not active_learning():
        print(f"Active Learning Iteration {i}")
        i += 1
    print("Color Matching Complete.")

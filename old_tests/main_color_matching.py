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
        filename = "color_matching_tiprack.json"
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
            volume: float) -> None:
        """
        Adds a color to the plate at the specified well.

        :param color_slot: The slot of the color reservoir.
        :param plate_well: The well of the plate to add the color to.
        :param volume: The volume of the color to add.

        :raises ValueError: If the well is already full.
        """
        global tiprack_state
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
    return 

    def active_learning() -> None:
        """
        The main loop of the protocol, which uses active learning to determine the best color match.
        """
        raise NotImplementedError

    # Main Loop
    i = 0
    while not active_learning():
        print(f"Active Learning Iteration {i}")
        i += 1
    print("Color Matching Complete.")
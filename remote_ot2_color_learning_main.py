import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from opentrons import protocol_api
import time 

color_slots = ['7','8','9']
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

metadata = {
    'protocolName': 'Blink Lights',
    'author': 'CMU Automated Lab',
    'description': 'Blink the lights.',
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

    def get_filename(filename: str) -> str:
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
            with open(get_filename('color_matching_tiprack.json'), 'r') as f:
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

    def pick_up_tip() -> None:
        """
        Picks up a tip from the tip rack.
        """
        global tiprack_state
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

    def return_tip() -> None:
        """
        Returns the tip to the tip rack.
        """
        pipette.drop_tip()


    ### CALLABLE FUNCTIONS ###
    def blink_lights(args: Dict[str, Any]) -> None:
        """
        Blink the lights on and off a number of times equal to the int in ./args.json
        times
        """
        # Check if 'num_blinks' is in args and is an integer
        if 'num_blinks' not in args:
            protocol.comment("num_blocks not found in args.json")
            return
        try:
            num_blinks = int(args['num_blinks'])
        except ValueError:
            protocol.comment("num_blocks is not an int")
            return
        
        protocol.comment(f"Blinking lights {num_blinks} times.")

        # Blink the lights
        for i in range(num_blinks):
            protocol.set_rail_lights(on=True)
            time.sleep(0.5)
            protocol.set_rail_lights(on=False)
    
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

    def close() -> None:
        """
        Closes the protocol, saving the state of the tip rack.
        """
        global tiprack_state, run_flag
        if  protocol.is_simulating():
            # don't save tiprack state in simulation
            return
        with open(get_filename('color_matching_tiprack.json'), 'w') as f:
            json.dump(tiprack_state, fp=f)

        run_flag = False
        protocol.comment("Protocol closed.")


    ### MAIN PROTOCOL ###

    #plate_type = get_plate_type()
    plate_type = "corning_96_wellplate_360ul_flat" # TODO: Remove this line when get_plate_type is implemented 
    #colors, plate, pipette, tiprack_state, off_deck_tipracks = setup(plate_type)
    # Wait for the json to change

    run_flag = True
    while run_flag:
        try:
            with open(get_filename('args.json'), 'r') as f:
                data: Dict[str, Any] = json.load(f)
        except FileNotFoundError:
            protocol.comment(f"{get_filename("args.json")} not found. Waiting...")
            time.sleep(1)
            continue
        except json.JSONDecodeError:
            protocol.comment("args.json is not valid JSON. Waiting...")
            time.sleep(1)
            continue
        except Exception as e:
            protocol.comment(f"Unexpected error: {e}. Waiting...")
            time.sleep(1)
            continue

        if "is_updated" not in data:
            protocol.comment("is_updated not found in args.json. Waiting...")
            time.sleep(1)
            continue

        if not data["is_updated"]:
            time.sleep(5)

        protocol.comment("args.json is updated. Running commands...")

        # At this point, we have a valid JSON file and is_updated is True
        # Now we must (a) set is_updated to False, and (b) run all the commands in the JSON file
        # A sample JSON file is:
        # {
        #     "is_updated": true,
        #     "actions": {
        #         "blink_lights": {
        #             "num_blinks": 5
        #          }
        #     }
        # }
        #
        # Note that the keys in "actions" are the names of the functions to call, and the values are the arguments to pass to those functions.

        actions: List[Tuple[str, Dict[str, Any]]] = data.get("actions", {})
        for action in actions:
            action_name, args = action
            if action_name in globals():
                func = globals()[action_name]
                if callable(func):
                    func(args)
                else:
                    protocol.comment(f"{action_name} is not callable.")

            else:
                protocol.comment(f"{action_name} not found in globals.")

        # Set is_updated to False
        data["is_updated"] = False
        # Remove the actions key
        data.pop("actions", None)

        # Write the updated JSON back to the file
        with open(get_filename('args.json'), 'w') as f:
            json.dump(data, f)
        protocol.comment("args.json updated. Waiting for next update...")
        protocol.comment("Ready")
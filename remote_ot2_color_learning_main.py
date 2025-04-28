import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from opentrons import protocol_api
import time 

color_slots = ['4','5','6','7','8','9','10','11']
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

metadata = {
    'protocolName': 'Blink Lights',
    'author': 'CMU Automated Lab',
    'description': 'Blink the lights.',
}

requirements = {'robotType': 'OT-2', 'apiLevel': '2.19'}

class WellFullError(Exception):
    """Exception raised when a well is full."""
    pass

class TiprackEmptyError(Exception):
    """Exception raised when the tip rack is empty."""
    pass

def run(protocol: protocol_api.ProtocolContext) -> None:
    """Defines the testing protocol."""
    protocol.comment("Start of run.")
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
        #protocol.comment(f"output file path = {output_file_destination_path}")
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

        # Check ./color_matching_tiprack.jsonx exists, if not make it and assume full rack
        try:
            with open(get_filename('color_matching_tiprack.jsonx'), 'r') as f:
                tiprack_state = json.load(f)
        except FileNotFoundError:
            protocol.comment(f"{get_filename('color_matching_tiprack.jsonx')} not found. Assuming full rack.")
            tiprack_state = [True] * 96
        except json.JSONDecodeError:
            protocol.comment(f"{get_filename('color_matching_tiprack.jsonx')} is not valid JSON. Assuming full rack.")
            protocol.comment(f"(The file had the following contents: {f.read()})")
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
            #protocol.comment("No tips left in the tip rack, switching to new rack.")
            #on_deck_position = pipette.tip_racks[0].parent
            #new_tiprack = off_deck_tipracks.pop()
            #protocol.move_labware(labware=pipette.tip_racks[0], new_location=protocol_api.OFF_DECK)
            #protocol.move_labware(labware=new_tiprack, new_location=on_deck_position)
            #pipette.tip_racks[0] = new_tiprack
            #next_well = 0
            #tiprack_state = [True] * 96

            # The above code only works via the OT2 Server GUI, not via the CLI.
            # So we will just raise an error instead.
            raise TiprackEmptyError("No tips left in the tip rack.")

        pipette.pick_up_tip(location=pipette.tip_racks[0].well(next_well))
        tiprack_state[next_well] = False

    def return_tip() -> None:
        """
        Returns the tip to the tip rack.
        """
        pipette.drop_tip()


    ### CALLABLE FUNCTIONS ###
    def blink_lights(num_blinks: int) -> None:
        """
        Blink the lights on and off a number of times equal to the int in ./args.jsonx
        times
        """
        protocol.comment(f"Blinking lights {num_blinks} times.")

        # Blink the lights
        for i in range(num_blinks):
            protocol.set_rail_lights(on=True)
            time.sleep(0.5)
            protocol.set_rail_lights(on=False)
            time.sleep(0.5)

    def turn_on_lights() -> None:
        """
        Turns on the lights.
        """
        protocol.comment("Turning on lights.")
        protocol.set_rail_lights(on=True)

    def turn_off_lights() -> None:
        """
        Turns off the lights.
        """
        protocol.comment("Turning off lights.")
        protocol.set_rail_lights(on=False)

    def refresh_tiprack() -> None:
        """
        Resets the tip rack state to all tips available.
        """
        global tiprack_state
        protocol.comment("Refreshing tip rack.")
        tiprack_state = [True] * 96
        protocol.comment("Tip rack refreshed.")

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
            raise WellFullError("Cannot add color to well; well is full.")

        pick_up_tip()
        pipette.aspirate(volume, colors[color_slot])
        pipette.touch_tip(plate.labware[plate_well], v_offset=95, radius=0) # necessary to avoid crashing against the large adapter
        pipette.dispense(volume, plate.labware[plate_well].bottom(z=81))

        plate.wells[plate_well].volume += volume

        # Quick mix (has to be manual because the default mix function doesn't work with the large adapter)
        pipette.aspirate(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.dispense(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.aspirate(volume/2, plate.labware[plate_well].bottom(z=81))
        pipette.dispense(volume/2, plate.labware[plate_well].bottom(z=81))

        # Blowout the remaining liquid in the pipette
        pipette.blow_out(plate.labware[plate_well].bottom(z=95))

        return_tip()

    def calibrate_96_well_plate() -> None:
        """
        Picks up a tip and moves it to A1 of the plate, pauses for 10 seconds,
        then moves it to H12 of the plate, pauses for 10 seconds, and then returns the tip.
        """
        global tiprack_state
        pick_up_tip()
        pipette.touch_tip(plate.labware['A1'], v_offset=95, radius=0)
        pipette.move_to(plate.labware['A1'].bottom(z=81))
        time.sleep(10)
        
        pipette.touch_tip(plate.labware['H12'], v_offset=95, radius=0)
        pipette.move_to(plate.labware['H12'].bottom(z=81))
        time.sleep(10)
        return_tip()

    def close() -> None:
        """
        Closes the protocol, saving the state of the tip rack.
        """
        global tiprack_state, run_flag
        if  protocol.is_simulating():
            # don't save tiprack state in simulation
            return
        with open(get_filename('color_matching_tiprack.jsonx'), 'w') as f:
            json.dump(tiprack_state, fp=f)

        run_flag = False
        protocol.comment("Protocol closed.")


    ### MAIN PROTOCOL ###

    #plate_type = get_plate_type()
    plate_type = "corning_96_wellplate_360ul_flat" # TODO: Remove this line when get_plate_type is implemented 
    global tiprack_state, run_flag
    protocol.comment("Loading labware and instruments...")
    colors, plate, pipette, tiprack_state, off_deck_tipracks = setup(plate_type)
    # Wait for the json to change

    run_flag = True
    protocol.comment("Ready")
    while run_flag:
        try:
            with open(get_filename('args.jsonx'), 'r') as f:
                data: Dict[str, Any] = json.load(f)
        except FileNotFoundError:
            protocol.comment(f"{get_filename('args.jsonx')} not found. Waiting...")
            time.sleep(1)
            continue
        except json.JSONDecodeError:
            protocol.comment("args.jsonx is not valid JSON. Waiting...")
            time.sleep(1)
            continue
        except Exception as e:
            protocol.comment(f"Unexpected error: {e}. Waiting...")
            time.sleep(1)
            continue

        if "is_updated" not in data:
            protocol.comment("is_updated not found in args.jsonx. Waiting...")
            time.sleep(1)
            continue

        if not data["is_updated"]:
            time.sleep(5)
            continue

        protocol.comment("args.jsonx is updated. Running commands...")

        # At this point, we have a valid JSON file and is_updated is True
        # Now we must (a) set is_updated to False, and (b) run all the commands in the JSON file
        # A sample JSON file is:
        # {
        #     "is_updated": true,
        #     "actions": [
        #         {
        #            "blink_lights": {
        #               "num_blinks": 5
        #            }
        #         },
        #         {
        #            "add_color": {
        #               "color_slot": "7",
        #               "plate_well": "A1",
        #               "volume": 100
        #            }
        #         },
        #     ]
        # }
        #
        # Note that the keys in "actions" are the names of the functions to call, and the values are the arguments to pass to those functions.

        actions: List[Dict[str, Dict[str, Any]]] = data.get("actions", {})
        for action in actions:
            for subaction_name, subaction_args in action.items():
                protocol.comment(f"Running {subaction_name} with args: {subaction_args}")
                #if subaction_name in globals():
                #    func = globals()[subaction_name]
                #    if callable(func):
                #        # Call the function with the arguments
                #        func(subaction_args)
                #    else:
                #        protocol.comment(f"{subaction_name} is not callable.")
                #else:
                #    protocol.comment(f"{subaction_name} not found in globals.")
                if subaction_name == "blink_lights":
                    blink_lights(subaction_args['num_blinks'])
                elif subaction_name == "turn_on_lights":
                    turn_on_lights()
                elif subaction_name == "turn_off_lights":
                    turn_off_lights()
                elif subaction_name == "refresh_tiprack":
                    refresh_tiprack()
                elif subaction_name == "add_color":
                    try:
                        add_color(subaction_args["color_slot"], subaction_args["plate_well"], subaction_args["volume"])
                    except WellFullError as e:
                        return failed_to_run_actions(e)
                    except TiprackEmptyError as e:
                        return failed_to_run_actions(e)
                elif subaction_name == "calibrate_96_well_plate":
                    calibrate_96_well_plate()
                elif subaction_name == "close":
                    close()
                    break
                else:
                    protocol.comment(f"{subaction_name} not found in defined commands.")

        # Set is_updated to False
        data["is_updated"] = False
        # Remove the actions key
        data.pop("actions", None)

        # Write the updated JSON back to the file
        with open(get_filename('args.jsonx'), 'w') as f:
            json.dump(data, f)
        protocol.comment("args.jsonx updated. Waiting for next update...")
        protocol.comment("Ready")


    def failed_to_run_actions(e: str) -> None:
        """
        Handle the case where the robot fails to run actions.

        :param e: The error message.
        """
        # Set is_updated to False
        data["is_updated"] = False
        # Remove the actions key
        data.pop("actions", None)

        # Write the updated JSON back to the file
        with open(get_filename('args.jsonx'), 'w') as f:
            json.dump(data, f)
        protocol.comment("args.jsonx updated. Waiting for next update...")
        protocol.comment(f"Error: {e}. Waiting for next update...")
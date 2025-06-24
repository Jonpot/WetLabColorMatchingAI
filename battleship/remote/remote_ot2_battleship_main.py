import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from opentrons import protocol_api
import time

ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

metadata = {
    'protocolName': 'Battleship Remote OT-2 Protocol',
    'author': 'CMU Automated Lab',
    'description': 'Control the OT-2 robot for the Battleship game.',
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

    def setup(plate_type: str = "corning_96_wellplate_360ul_flat",
              plate_1_slot: str = "1",
              plate_2_slot: str = "4",
              ammo_slot: str = "2",
              tiprack_slot: str = "3",
              ocean_fluid_slot: str = "5",
              ship_fluid_slot: str = "6",
              default_volume: int = 50) -> Tuple[protocol_api.Well, protocol_api.Well, protocol_api.Well,
                                                  Plate, Plate, protocol_api.InstrumentContext,
                                                  List[bool],
                                                  List[protocol_api.Labware]]:
        """
        Loads labware and instruments for the protocol.

        :param plate_type: The type of plate to use, as per the Opentrons API.
        """
        tipracks: List[protocol_api.Labware] = [protocol.load_labware('opentrons_96_tiprack_300ul', location=tiprack_slot)]

        # Some tips may be missing, so we need to update the current state of the tip rack from
        # the file. This is necessary to avoid the robot trying to use tips that are not present.

        # Check ./tiprack_state.jsonx exists, if not make it and assume full rack
        try:
            with open(get_filename('tiprack_state.jsonx'), 'r') as f:
                tiprack_state: List[bool] = json.load(f)
        except FileNotFoundError:
            protocol.comment(f"{get_filename('tiprack_state.jsonx')} not found. Assuming full rack.")
            tiprack_state = [True] * 96
        except json.JSONDecodeError:
            protocol.comment(f"{get_filename('tiprack_state.jsonx')} is not valid JSON. Assuming full rack.")
            protocol.comment(f"(The file had the following contents: {f.read()})")
            tiprack_state = [True] * 96

        ammo = protocol.load_labware('nest_12_reservoir_15ml', location=ammo_slot)['A1']
        ocean_fluid = protocol.load_labware('nest_12_reservoir_15ml', location=ocean_fluid_slot)['A1']
        ship_fluid = protocol.load_labware('nest_12_reservoir_15ml', location=ship_fluid_slot)['A1']

        plate_1_labware = protocol.load_labware(plate_type, label="Battleship Plate", location=plate_1_slot)
        plate_1 = Plate(plate_1_labware, len(plate_1_labware.rows()), len(plate_1_labware.columns()), plate_1_labware.wells()[0].max_volume)

        plate_2_labware = protocol.load_labware(plate_type, label="Battleship Plate 2", location=plate_2_slot)
        plate_2 = Plate(plate_2_labware, len(plate_2_labware.rows()), len(plate_2_labware.columns()), plate_2_labware.wells()[0].max_volume)

        pipette = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=tipracks)

        off_deck_tipracks: List[protocol_api.Labware] = []
        for _ in range(10): # arbitrarily high number of tip boxes
            # these tip boxes will be replaced as needed
            off_deck_tipracks.append(protocol.load_labware('opentrons_96_tiprack_300ul', location=protocol_api.OFF_DECK))

        return ammo, ocean_fluid, ship_fluid, plate_1, plate_2, pipette, tiprack_state, off_deck_tipracks

    def pick_up_tip(tip_ID: str = None) -> None:
        """
        Picks up a tip from the tip rack.
        """
        global tiprack_state, reduced_tips_info

        if reduced_tips_info is not None:
            if tip_ID not in reduced_tips_info:
                try:
                    tip_ID_well = tiprack_state.index(True)
                except ValueError:
                    raise TiprackEmptyError(f"No tips left in the tip rack to assign for {tip_ID}.")
                reduced_tips_info[tip_ID] = tip_ID_well
                protocol.comment(f"Using tip {tip_ID_well} for color slot {tip_ID}.")
                tiprack_state[tip_ID_well] = False

            # At this point, this color slot has a dedicated tip assigned to it.
            # Pick up this tip
            protocol.comment(f"Picking up tip {reduced_tips_info[tip_ID]} for color slot {tip_ID}. Exact arg: {pipette.tip_racks[0].wells()[reduced_tips_info[tip_ID]]}")
            pipette.pick_up_tip(location=pipette.tip_racks[0].wells()[reduced_tips_info[tip_ID]])
            return
            

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

    def return_tip(tip_ID: str = None) -> None:
        """
        Returns the tip to the tip rack.
        """
        global reduced_tips_info

        if reduced_tips_info is not None:
            # Then we need to return this tip back to the tip rack
            if tip_ID not in reduced_tips_info:
                protocol.comment(f"Something is wrong. Tip {tip_ID} is not in reduced_tips_info: {reduced_tips_info}, but then I don't know how I got this tip.")
                pipette.drop_tip()
                return 
            
            tip_ID_well = reduced_tips_info[tip_ID]
            protocol.comment(f"Returning tip to tipbox slot {tip_ID_well}.")
            pipette.return_tip()
        else:
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
        Resets the tip rack state to all tips available and resets pipette tracking.
        """
        global tiprack_state
        protocol.comment("Refreshing tip rack...")
        tiprack_state = [True] * 96
        pipette.reset_tipracks()
        protocol.comment("Tip rack refreshed.")

    def fire_missile(
            plate_idx: int,
            plate_well: str,
            volume: float) -> None:
        """
        Fires a missile to the plate at the specified well.

        :param plate_well: The well of the plate to add the color to.
        :param volume: The volume of the color to add.

        :raises ValueError: If the well is already full.
        """
        global tiprack_state, reduced_tips_info

        if plate_idx not in [1, 2]:
            raise ValueError("Invalid plate number. Must be 1 or 2.")
        if plate_idx == 1:
            plate = plate_1
        elif plate_idx == 2:
            plate = plate_2

        if volume + plate.wells[plate_well].volume > plate.wells[plate_well].max_volume:
            raise WellFullError("Cannot add color to well; well is full.")

        pick_up_tip(tip_ID="missile")
        pipette.aspirate(volume, ammo)
        pipette.touch_tip(plate.labware[plate_well], v_offset=15, radius=0) # necessary to avoid crashing against the large adapter
        pipette.dispense(volume, plate.labware[plate_well].bottom(z=2.5))

        plate.wells[plate_well].volume += volume

        # Blowout the remaining liquid in the pipette
        pipette.blow_out(plate.labware[plate_well].bottom(z=15))

        return_tip(tip_ID="missile")

        mix(plate_idx, plate_well, min(200,volume*2), 3)
    
    def mix(
            plate_idx: int,
            plate_well: str | int,
            volume: float,
            repetitions: int) -> None:
        """
        Mixes the contents of a well.

        :param plate_well: The well of the plate to mix.
        :param volume: The volume to mix.
        :param repetitions: The number of times to mix.
        """
        global tiprack_state, reduced_tips_info

        if plate_idx not in [1, 2]:
            raise ValueError("Invalid plate number. Must be 1 or 2.")
        if plate_idx == 1:
            plate = plate_1
        elif plate_idx == 2:
            plate = plate_2

        pick_up_tip("mix") # dedicated tip for mixing

        pipette.touch_tip(plate.labware[plate_well], v_offset=95, radius=0) # necessary to avoid crashing against the large adapter
        # Quick mix (has to be manual because the default mix function doesn't work with the large adapter)
        for _ in range(repetitions):
            pipette.aspirate(volume, plate.labware[plate_well].bottom(z=2.5))
            pipette.dispense(volume, plate.labware[plate_well].bottom(z=2.5))

        # Blowout the remaining liquid in the pipette
        pipette.blow_out(plate.labware[plate_well].bottom(z=15))

        return_tip("mix")

    def _place_liquid(
            liquid: protocol_api.Well,
            plate_idx: int,
            wells: List[str]) -> None:
        """Helper to dispense liquid into many wells efficiently."""
        if plate_idx not in [1, 2]:
            raise ValueError("Invalid plate number. Must be 1 or 2.")
        plate = plate_1 if plate_idx == 1 else plate_2

        pick_up_tip("placement")
        remaining = 0
        for well in wells:
            if remaining < default_volume:
                pipette.aspirate(1000, liquid)
                remaining = 1000
            pipette.dispense(default_volume, plate.labware[well].bottom(z=2.5))
            remaining -= default_volume
            plate.wells[well].volume += default_volume
        pipette.blow_out(liquid.top())
        return_tip("placement")

    def place_water_in_wells(plate_idx: int, wells: List[str]) -> None:
        _place_liquid(ocean_fluid, plate_idx, wells)

    def place_ships_in_wells(plate_idx: int, wells: List[str]) -> None:
        _place_liquid(ship_fluid, plate_idx, wells)



    def calibrate_96_well_plate() -> None:
        """
        Picks up a tip and moves it to A1 of the plate, pauses for 10 seconds,
        then moves it to H12 of the plate, pauses for 10 seconds, and then returns the tip.
        """
        global tiprack_state
        pick_up_tip()
        pipette.touch_tip(plate_1.labware['A1'], radius=0)
        pipette.move_to(plate_1.labware['A1'].bottom())
        time.sleep(10)
        
        pipette.touch_tip(plate_1.labware['H12'], radius=0)
        pipette.move_to(plate_1.labware['H12'].bottom())
        time.sleep(10)
        return_tip()

    def close() -> None:
        """
        Closes the protocol, saving the state of the tip rack.
        """
        global tiprack_state, run_flag, reduced_tips_info
        if  protocol.is_simulating():
            # don't save tiprack state in simulation
            return
        
        # if using reduced tips, move all the tips to trash
        if reduced_tips_info is not None:
            for tip_ID, tip in reduced_tips_info.items():
                protocol.comment(f"Returning tip {tip} to trash for tip ID {tip_ID}.")
                pipette.pick_up_tip(location=pipette.tip_racks[0].wells()[tip])
                pipette.drop_tip()

        with open(get_filename('tiprack_state.jsonx'), 'w') as f:
            json.dump(tiprack_state, fp=f)

        run_flag = False
        protocol.comment("Protocol closed.")


    ### MAIN PROTOCOL ###

    plate_type = "corning_96_wellplate_360ul_flat"
    global tiprack_state, run_flag, reduced_tips_info
    reduced_tips_info = {}
    # Wait for the json to change

    run_flag = True

    # Check for special tiprack information
    try:
        with open(get_filename('args.jsonx'), 'r') as f:
            data: Dict[str, Any] = json.load(f)
            if "reduced_tips_info" in data:
                n = data["reduced_tips_info"]
                protocol.comment(f"Reduced tips info found. {n+1} tips will be used, 1 for each color and 1 for mixing.")
                # ensure that n+1 tips are available in the tip rack state
                if len([x for x in tiprack_state if x]) < n + 1:
                    protocol.comment(f"Not enough tips available in the tip rack. {n+1} tips are needed, but only {len([x for x in tiprack_state if x])} are available.")
                    raise TiprackEmptyError("Not enough tips available in the tip rack.")
                reduced_tips_info = {}


    except FileNotFoundError:
        protocol.comment(f"{get_filename('args.jsonx')} not found. Assuming regular tip usage.")
    except json.JSONDecodeError:
        protocol.comment(f"{get_filename('args.jsonx')} is not valid JSON. Assuming regular tip usage.")
        protocol.comment(f"(The file had the following contents: {f.read()})")
    except Exception as e:
        protocol.comment(f"Unexpected error: {e}. Assuming regular tip usage.")
    
    plate_1_slot = data.get("plate_1_slot", "1")
    plate_2_slot = data.get("plate_2_slot", "4")
    ammo_slot = data.get("ammo_slot", "2")
    tiprack_slot = data.get("tiprack_slot", "3")
    ocean_fluid_slot = data.get("ocean_fluid_slot", "5")
    ship_fluid_slot = data.get("ship_fluid_slot", "6")
    missile_volume = data.get("missile_volume", 50)
    default_volume = data.get("default_volume", 50)
    protocol.comment("Loading labware and instruments...")
    ammo, ocean_fluid, ship_fluid, plate_1, plate_2, pipette, tiprack_state, off_deck_tipracks = setup(plate_type, plate_1_slot, plate_2_slot, ammo_slot, tiprack_slot, ocean_fluid_slot, ship_fluid_slot, default_volume)
    
    
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
        #               "tip_ID": "7",
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
                elif subaction_name == "fire_missile":
                    try:
                        fire_missile(subaction_args["plate_idx"], subaction_args["plate_well"], missile_volume)
                    except WellFullError as e:
                        return failed_to_run_actions(e)
                    except TiprackEmptyError as e:
                        return failed_to_run_actions(e)
                elif subaction_name == "place_water_in_wells":
                    place_water_in_wells(subaction_args["plate_idx"], subaction_args["wells"])
                elif subaction_name == "place_ships_in_wells":
                    place_ships_in_wells(subaction_args["plate_idx"], subaction_args["wells"])
                elif subaction_name == "calibrate_96_well_plate":
                    calibrate_96_well_plate()
                elif subaction_name == "close":
                    close()
                    break
                else:
                    protocol.comment(f"{subaction_name} not found in defined commands.")

        # Move the pipette to the trash to avoid covering the camera
        if protocol.is_simulating():
            protocol.comment("Simulating: moving pipette to trash.")
        else:
            protocol.comment("Moving pipette to trash.")
            pipette.move_to(pipette.trash_container.top())

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
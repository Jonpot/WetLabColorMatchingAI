import json
from pathlib import Path
import sys
from opentrons import protocol_api

metadata = {
    'protocolName': '[Testing] Near Empty Tiprack',
    'author': 'CMU Automated Lab',
    'description': 'Run this if you want to simulate a tiprack with 1 tip left.',
}

requirements = {'robotType': 'OT-2', 'apiLevel': '2.19'}


def run(protocol: protocol_api.ProtocolContext) -> None:
    """Defines the testing protocol."""

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

    def reset_tiprack() -> None:
        """
        Resets the tiprack state to all tips being available.
        """
        if  protocol.is_simulating():
            # don't save tiprack state in simulation
            return
        tiprack_state = [False] * 96
        tiprack_state[-1] = True
        with open(get_filename(), 'w') as f:
            json.dump(tiprack_state, fp=f)

    reset_tiprack()
    return 
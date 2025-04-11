import json
from opentrons import protocol_api
import time 

metadata = {
    'protocolName': 'Simple Blink Lights',
    'author': 'CMU Automated Lab',
    'description': 'Blink the lights.',
}

requirements = {'robotType': 'OT-2', 'apiLevel': '2.19'}


def run(protocol: protocol_api.ProtocolContext) -> None:
    """Defines the testing protocol."""
    
    num_blinks = 5  # Default value
    protocol.comment(f"Blinking lights {num_blinks} times.")

    # Blink the lights
    for i in range(num_blinks):
        protocol.set_rail_lights(on=True)
        time.sleep(0.5)  # Light on for 0.5 seconds
        protocol.set_rail_lights(on=False)
        time.sleep(0.5)  # Light off for 0.5 seconds
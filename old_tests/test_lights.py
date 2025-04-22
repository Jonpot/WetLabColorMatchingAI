import json
from opentrons import protocol_api
import time 

metadata = {
    'protocolName': 'Blink Lights',
    'author': 'CMU Automated Lab',
    'description': 'Blink the lights.',
}

requirements = {'robotType': 'OT-2', 'apiLevel': '2.19'}


def run(protocol: protocol_api.ProtocolContext) -> None:
    """Defines the testing protocol."""

    # Blink the lights on and off a number of times equal to the int in ./args.json
    # times
    
    # Load the json
    with open('args.json', 'r') as f:
        data = json.load(f)

    if 'num_blinks' not in data:
        protocol.comment("num_blocks not found in args.json")
        return
    try:
        num_blinks = int(data['num_blinks'])
    except ValueError:
        protocol.comment("num_blocks is not an int")
        return
    
    protocol.comment(f"Blinking lights {num_blinks} times.")

    # Blink the lights
    for i in range(num_blinks):
        protocol.set_rail_lights(on=True)
        time.sleep(0.5)  # Light on for 0.5 seconds
        protocol.set_rail_lights(on=False)
        time.sleep(0.5)  # Light off for 0.5 seconds
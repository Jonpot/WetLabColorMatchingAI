from importlib import import_module

# Expose the OT2 utilities from the battleship package for backwards compatibility
ot2_utils = import_module('battleship.robot.ot2_utils')

__all__ = ['ot2_utils']

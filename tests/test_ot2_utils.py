import importlib
import sys
import types
import json
import tempfile
import unittest
from pathlib import Path

# Stub paramiko and scp so module can be imported without the packages
## this is a workaround for the fact that the packages are not installed in the test environment
paramiko_stub = types.ModuleType("paramiko")
paramiko_stub.SSHClient = object
paramiko_stub.AutoAddPolicy = object
class DummyRSAKey:
    @classmethod
    def from_private_key_file(cls, *args, **kwargs):
        return None
paramiko_stub.RSAKey = DummyRSAKey
sys.modules.setdefault("paramiko", paramiko_stub)

scp_stub = types.ModuleType("scp")
scp_stub.SCPClient = object
sys.modules.setdefault("scp", scp_stub)

ot2_utils = importlib.import_module("robot.ot2_utils")


class OT2UtilsTests(unittest.TestCase):
    def test_get_plate_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "camera/calibration.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps({"plate_type": "24"}))
            plate = ot2_utils.get_plate_type(str(cfg_path))
            self.assertEqual(plate, "corning_24_wellplate_3.4ml_flat")

    def test_get_plate_type_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.json"
            plate = ot2_utils.get_plate_type(str(missing_path))
            self.assertEqual(plate, "corning_96_wellplate_360ul_flat")


if __name__ == "__main__":
    unittest.main()

"""Manual test script for the active learning pipeline.

This module is not executed during automated tests.  Its contents are wrapped
in a ``__main__`` guard so ``unittest`` discovery does not attempt to import
external dependencies like ``paramiko``.
"""

if __name__ == "__main__":
    from robot.ot2_utils import OT2Manager, TiprackEmptyError
    from camera.camera_w_calibration import PlateProcessor
    from active_learning.color_learning import ColorLearningOptimizer
    from main_active_learning import active_learn_row

    color_slots = ["7", "8", "9", "11"]

    optimizer = ColorLearningOptimizer(dye_count=len(color_slots),
                                       tolerance=80)

    row = "A"

    robot = OT2Manager(
        hostname="172.26.192.201",
        username="root",
        key_filename="secret/ot2_ssh_key_remote",
        password=None,
        reduced_tips_info=len(color_slots),
        virtual_mode=False,
    )

    CAM_INDEX = 2

    processor = PlateProcessor()

    robot.add_turn_on_lights_action()
    robot.execute_actions_on_remote()

    measured_plate = processor.process_image(cam_index=CAM_INDEX)
    target_color = measured_plate[0][0]

    def log_cb(msg: str) -> None:
        print(msg)

    active_learn_row(
        robot=robot,
        processor=processor,
        optimizer=optimizer,
        row_letter=row,
        target_color=target_color,
        color_slots=color_slots,
        cam_index=CAM_INDEX,
        log_cb=log_cb,
    )

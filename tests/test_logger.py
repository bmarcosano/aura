from aura.logger import AuraLogger


def test_static_events_render_without_unpacking_strings():
    logger = AuraLogger()
    logger.log_static("calibration", "camera calibration loaded")
    logger.log("Ready", "System is ready")
    assert logger.render_events().shape == (420, 560, 3)
    assert logger.render().shape == (420, 860, 3)

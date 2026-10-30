from aura.fusion import AuraFusionEngine


def test_missing_view_is_not_reported_as_optimal():
    result = AuraFusionEngine().evaluate_posture(None, None)
    assert result["state"] == "INSUFFICIENT_DATA"


def test_single_view_reports_partial_status():
    # A minimal landmark container with adequate visibility and near-level joints.
    class Landmark:
        def __init__(self, x=0.5, y=0.5, visibility=1.0):
            self.x, self.y, self.visibility = x, y, visibility

    class Pose:
        def __init__(self):
            values = [Landmark() for _ in range(33)]
            values[11], values[12] = Landmark(0.4, 0.4), Landmark(0.6, 0.4)
            values[23], values[24] = Landmark(0.4, 0.7), Landmark(0.6, 0.7)
            values[25], values[26] = Landmark(0.4, 0.8), Landmark(0.6, 0.8)
            self.landmark = values

    class Results:
        pose_landmarks = Pose()

    result = AuraFusionEngine().evaluate_posture(Results(), None)
    assert result["state"] == "PARTIAL"
    assert result["weighted_score"] == 2.0

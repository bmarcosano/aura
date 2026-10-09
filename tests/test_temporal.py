"""
Pytest module for the DTWTemporalAnalyzer.
Run these tests from the terminal using: pytest test_temporal.py -v
"""

import pytest

from aura.temporal import DTWTemporalAnalyzer

# ========================================================
# MOCK DATA SETUP
# ========================================================
# GOLDEN STANDARD: Perfect squat trajectory (180° -> 70° -> 180°)
GOLDEN_REP = list(range(180, 69, -5)) + list(range(70, 181, 5))

# SLOW USER: Perfect form, but exactly half the speed (frames duplicated)
SLOW_GOOD_REP = [val for val in GOLDEN_REP for _ in (0, 1)]

# POOR USER: Shallow squat (stops at 110°), awkward pause, rushed ascent
BAD_REP = list(range(180, 109, -5)) + [110] * 10 + list(range(110, 181, 10))


@pytest.fixture
def analyzer():
    """
    Pytest fixture to initialize the DTWTemporalAnalyzer and load the
    Golden Standard before each test function runs. This keeps tests DRY.
    """
    dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)
    dtw_analyzer.load_golden_standard(GOLDEN_REP)
    return dtw_analyzer


def test_slow_perfect_execution(analyzer):
    """
    Test that an identical but slower movement scores perfectly.
    DTW should compress the temporal elasticity and recognize the perfect form.
    """
    for angle in SLOW_GOOD_REP:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()

    assert result.get("status") == "SUCCESS"
    assert result.get("score") == 100.0
    assert result.get("feedback") == "EXCELLENT"
    assert result.get("frames_analyzed") == len(SLOW_GOOD_REP)


def test_poor_execution(analyzer):
    """
    Test that a poor, shortened execution results in a heavily penalized score.
    """
    for angle in BAD_REP:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()

    assert result.get("status") == "SUCCESS"
    assert result.get("score") < 70.0
    assert result.get("feedback") in [
        "FAIR - Form breaking down",
        "POOR - Biomechanical risk",
    ]


def test_live_pacing_feedback(analyzer):
    """
    Test the real-time pacing feedback logic (too fast, too slow, optimal).
    """
    # 1. Test IDLE state when buffer is empty
    assert analyzer.check_live_pacing() == "PACE: IDLE"

    # 2. Test TOO FAST scenario (feeding too few frames compared to golden reference)
    for angle in GOLDEN_REP[:10]:
        analyzer.add_user_frame(angle)
    # With a small buffer and ratio < 0.75 (after bypassing initial short check), it should flag as too fast
    # Let's feed enough frames to pass the threshold but keep the ratio low
    analyzer.reset_user_buffer()
    fast_rep = GOLDEN_REP[: len(GOLDEN_REP) // 3]
    for angle in fast_rep:
        analyzer.add_user_frame(angle)

    pacing_fast = analyzer.check_live_pacing()
    assert "TOO FAST" in pacing_fast

    # 3. Test TOO SLOW scenario (feeding an oversized buffer)
    analyzer.reset_user_buffer()
    very_slow_rep = GOLDEN_REP * 2  # Double the frames
    for angle in very_slow_rep:
        analyzer.add_user_frame(angle)

    pacing_slow = analyzer.check_live_pacing()
    assert "TOO SLOW" in pacing_slow


def test_empty_golden_standard_error():
    """
    Test the failsafe mechanism when trying to evaluate a user without
    a loaded reference trajectory.
    """
    empty_analyzer = DTWTemporalAnalyzer()
    empty_analyzer.add_user_frame(180)

    result = empty_analyzer.evaluate_technique()

    assert "error" in result
    assert result["error"] == "No Golden Standard loaded."
    assert result["score"] == 0


def test_user_buffer_too_short(analyzer):
    """
    Test the failsafe mechanism when the user array is too small to be meaningful.
    """
    for angle in [180, 175, 170]:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()

    assert "error" in result
    assert result["error"] == "User movement too short for analysis."
    assert result["score"] == 0

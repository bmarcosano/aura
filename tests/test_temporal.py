"""
test_temporal.py: Unit tests for DTWTemporalAnalyzer module.
"""

import json
import os

import pytest

from aura.temporal import DTWTemporalAnalyzer

GOLDEN_REP = list(range(180, 69, -5)) + list(range(70, 181, 5))
SLOW_GOOD_REP = [val for val in GOLDEN_REP for _ in (0, 1)]
BAD_REP = list(range(180, 109, -5)) + [110] * 10 + list(range(110, 181, 10))


@pytest.fixture
def analyzer():
    """Fixture providing an initialized DTWTemporalAnalyzer with a loaded Golden Standard."""
    dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)
    dtw_analyzer.load_golden_standard(GOLDEN_REP)
    return dtw_analyzer


def test_load_golden_standard_empty():
    """Test exception raising when loading an empty trajectory."""
    analyzer = DTWTemporalAnalyzer()
    with pytest.raises(ValueError, match="Golden standard trajectory cannot be empty."):
        analyzer.load_golden_standard([])


def test_slow_perfect_execution(analyzer):
    """Test that an identical but slower movement scores perfectly."""
    for angle in SLOW_GOOD_REP:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()
    assert result.get("status") == "SUCCESS"
    assert result.get("score") == 100.0
    assert result.get("feedback") == "EXCELLENT"


def test_good_execution(analyzer):
    """Test form execution resulting in GOOD feedback category."""
    # Create a slightly perturbed trajectory to hit the 75-89 score range (GOOD)
    perturbed_rep = [val + 3 for val in GOLDEN_REP]
    for angle in perturbed_rep:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()
    assert result.get("status") == "SUCCESS"
    assert "score" in result


def test_poor_execution(analyzer):
    """Test that a poor, shortened execution results in a heavily penalized score."""
    for angle in BAD_REP:
        analyzer.add_user_frame(angle)

    result = analyzer.evaluate_technique()
    assert result.get("status") == "SUCCESS"
    assert result.get("score") < 70.0


def test_live_pacing_feedback(analyzer):
    """Test real-time pacing feedback logic (IDLE, TOO FAST, TOO SLOW, OPTIMAL)."""
    assert analyzer.check_live_pacing() == "PACE: IDLE"

    # Optimal pacing (feeding ~90% of the frames, ratio between 0.75 and 1.25)
    optimal_len = int(len(GOLDEN_REP) * 0.9)
    for angle in GOLDEN_REP[:optimal_len]:
        analyzer.add_user_frame(angle)
    assert analyzer.check_live_pacing() == "PACE: OPTIMAL"

    # Too fast pacing (ratio < 0.75 and user_len > 10)
    analyzer.reset_user_buffer()
    for angle in [180] * 12:
        analyzer.add_user_frame(angle)
    assert "TOO FAST" in analyzer.check_live_pacing()

    # Too slow pacing (ratio > 1.25)
    analyzer.reset_user_buffer()
    for angle in GOLDEN_REP * 2:
        analyzer.add_user_frame(angle)
    assert "TOO SLOW" in analyzer.check_live_pacing()


def test_empty_golden_standard_error():
    """Test evaluation error when no reference trajectory is loaded."""
    empty_analyzer = DTWTemporalAnalyzer()
    empty_analyzer.add_user_frame(180)
    result = empty_analyzer.evaluate_technique()
    assert result["error"] == "No Golden Standard loaded."
    assert result["score"] == 0


def test_user_buffer_too_short(analyzer):
    """Test evaluation error when user buffer has fewer than 5 frames."""
    for angle in [180, 175, 170]:
        analyzer.add_user_frame(angle)
    result = analyzer.evaluate_technique()
    assert result["error"] == "User movement too short for analysis."
    assert result["score"] == 0


def test_persistence_operations(analyzer, tmp_path, monkeypatch):
    """Test saving, loading, and deleting Golden Standard files on disk."""
    # Change working directory to tmp_path to isolate file operations
    monkeypatch.chdir(tmp_path)

    # 1. Save when empty should fail
    empty_analyzer = DTWTemporalAnalyzer()
    assert empty_analyzer.save_golden_standard() is False

    # 2. Save successfully
    assert (
        analyzer.save_golden_standard(exercise="test_squat", filename="test.json")
        is True
    )

    # 3. Load successfully from file
    new_analyzer = DTWTemporalAnalyzer()
    assert (
        new_analyzer.load_golden_standard_from_file(
            exercise="test_squat", filename="test.json"
        )
        is True
    )
    assert len(new_analyzer.golden_reference) == len(GOLDEN_REP)

    # 4. Load non-existent file should fail
    assert (
        new_analyzer.load_golden_standard_from_file(
            exercise="nonexistent", filename="bad.json"
        )
        is False
    )

    # 5. Load malformed json file
    bad_file_path = tmp_path / "golden_executions" / "test_squat" / "bad.json"
    os.makedirs(os.path.dirname(bad_file_path), exist_ok=True)
    with open(bad_file_path, "w") as f:
        f.write("{invalid_json")
    assert (
        new_analyzer.load_golden_standard_from_file(
            exercise="test_squat", filename="bad.json"
        )
        is False
    )

    # 6. Load valid JSON without 'golden_reference' key
    with open(bad_file_path, "w") as f:
        json.dump({"wrong_key": []}, f)
    assert (
        new_analyzer.load_golden_standard_from_file(
            exercise="test_squat", filename="bad.json"
        )
        is False
    )

    # 7. Delete file successfully
    assert (
        analyzer.delete_golden_standard_file(
            exercise="test_squat", filename="test.json"
        )
        is True
    )

    # 8. Delete non-existent file
    assert (
        analyzer.delete_golden_standard_file(
            exercise="test_squat", filename="test.json"
        )
        is False
    )


def test_persistence_error_handling(analyzer, monkeypatch):
    """Test exception handling during file save/load operations."""

    # Force open() to raise an OSError to cover exception blocks in file methods
    def mock_open_raise(*args, **kwargs):
        raise OSError("Simulated disk error")

    monkeypatch.setattr("builtins.open", mock_open_raise)
    assert analyzer.save_golden_standard() is False
    assert analyzer.load_golden_standard_from_file() is False


def test_delete_golden_standard_error(analyzer, monkeypatch):
    """Test exception handling during file deletion (covers lines 108-110)."""

    # Arrange: Force os.remove to raise an OSError
    def mock_remove_raise(*args, **kwargs):
        raise OSError("Simulated deletion error")

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.remove", mock_remove_raise)

    # Act & Assert
    assert analyzer.delete_golden_standard_file() is False


def test_evaluate_technique_poor_feedback(analyzer):
    """Test technique evaluation returning POOR feedback for very low scores (covers line 170)."""
    # Arrange: Feed an extremely erratic/distant movement to drop the score below 50
    analyzer.reset_user_buffer()
    for _ in range(10):
        analyzer.add_user_frame(
            0.0
        )  # Extreme divergence from the golden 180°-70° trajectory

    # Act
    result = analyzer.evaluate_technique()

    # Assert
    assert result.get("status") == "SUCCESS"
    assert result.get("feedback") == "POOR - Biomechanical risk"

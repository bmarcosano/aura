import numpy as np

from aura.temporal import DTWTemporalAnalyzer


def test_dtw_identical_sequence_scores_100():
    analyzer = DTWTemporalAnalyzer(sensitivity=5.0)
    sequence = np.array([[170, 170, 90, 90]] * 6, dtype=float)
    analyzer.load_golden_standard(sequence)
    for row in sequence:
        analyzer.add_user_frame(row)
    result = analyzer.evaluate_technique()
    assert result["status"] == "SUCCESS"
    assert result["score"] == 100.0


def test_dtw_rejects_feature_dimension_mismatch():
    analyzer = DTWTemporalAnalyzer()
    analyzer.load_golden_standard([1, 2, 3, 4, 5])
    for _ in range(5):
        analyzer.add_user_frame([1, 2, 3, 4])
    result = analyzer.evaluate_technique()
    assert result["status"] == "INCOMPATIBLE_REFERENCE"


def test_nonfinite_user_frame_is_not_recorded():
    analyzer = DTWTemporalAnalyzer()
    analyzer.add_user_frame([1.0, np.nan, 3.0])
    assert analyzer.user_buffer == []

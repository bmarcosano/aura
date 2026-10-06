from __future__ import annotations

"""
Unit Tests for AURA State Machine & Mode Manager
Tests dynamic state transitions between single-view and mirror fusion modes.
"""


import pytest


class MockStateMachine:
    """Mock implementation of AURA State Machine for transition logic testing."""

    STATE_SINGLE = "SINGLE_MODE"
    STATE_CHECKING = "CHECKING"
    STATE_MIRROR = "MIRROR_MODE"

    def __init__(self):
        self.current_state = self.STATE_SINGLE
        self.multi_detector = None

    def update(self, mirror_detected: bool, split_ratio: float | None = None):
        """Updates the state based on sentinel triggers and sync analysis results."""
        if self.current_state == self.STATE_SINGLE and mirror_detected:
            self.current_state = self.STATE_CHECKING

        elif self.current_state == self.STATE_CHECKING:
            if split_ratio is not None:
                self.current_state = self.STATE_MIRROR
                self.multi_detector = f"Detector_Split_{split_ratio}"
            else:
                # Fallback back to single mode if synchronization fails
                self.current_state = self.STATE_SINGLE


@pytest.fixture
def state_machine():
    """Fixture providing a fresh state machine instance."""
    return MockStateMachine()


def test_initial_state(state_machine):
    """Test that the application boots up in SINGLE_MODE by default."""
    assert state_machine.current_state == MockStateMachine.STATE_SINGLE
    assert state_machine.multi_detector is None


def test_transition_to_checking(state_machine):
    """Test that detecting potential mirror blobs triggers the CHECKING state."""
    # Simulate motion sentinel detecting multiple blobs
    state_machine.update(mirror_detected=True)

    assert state_machine.current_state == MockStateMachine.STATE_CHECKING


def test_full_transition_to_mirror_mode(state_machine):
    """Test full successful transition pipeline from SINGLE to MIRROR_MODE via CHECKING."""
    # Step 1: Trigger checking state
    state_machine.update(mirror_detected=True)
    assert state_machine.current_state == MockStateMachine.STATE_CHECKING

    # Step 2: SyncAnalyzer confirms synchronization and returns split ratio
    state_machine.update(mirror_detected=True, split_ratio=0.55)

    assert state_machine.current_state == MockStateMachine.STATE_MIRROR
    assert state_machine.multi_detector == "Detector_Split_0.55"


def test_fallback_to_single_mode_on_sync_failure(state_machine):
    """Test that the system falls back to SINGLE_MODE if sync validation fails during checking."""
    # Trigger checking
    state_machine.update(mirror_detected=True)
    assert state_machine.current_state == MockStateMachine.STATE_CHECKING

    # Sync fails (split_ratio is None)
    state_machine.update(mirror_detected=False, split_ratio=None)

    assert state_machine.current_state == MockStateMachine.STATE_SINGLE
    assert state_machine.multi_detector is None

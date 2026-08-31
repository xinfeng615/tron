import numpy as np
import pytest

from examples.tron2.action_safety import ActionSafetyConfig, ActionSafetyError, ActionSafetyGate


def test_valid_action_clips_grippers():
    gate = ActionSafetyGate(ActionSafetyConfig(max_arm_delta_rad=0.1))
    state = np.zeros(16, dtype=np.float32)
    actions = np.zeros((2, 16), dtype=np.float32)
    actions[:, 7] = [-0.5, 0.5]
    actions[:, 15] = [1.5, 0.25]
    result = gate.validate(actions, state)
    np.testing.assert_allclose(result[:, 7], [0.0, 0.5])
    np.testing.assert_allclose(result[:, 15], [1.0, 0.25])


@pytest.mark.parametrize("shape", [(15,), (1, 18), (1, 1, 16)])
def test_rejects_wrong_shape(shape):
    gate = ActionSafetyGate(ActionSafetyConfig())
    with pytest.raises(ActionSafetyError, match="Expected actions"):
        gate.validate(np.zeros(shape), np.zeros(16))


def test_rejects_nonfinite_action():
    gate = ActionSafetyGate(ActionSafetyConfig())
    action = np.zeros((1, 16))
    action[0, 0] = np.nan
    with pytest.raises(ActionSafetyError, match="NaN or infinity"):
        gate.validate(action, np.zeros(16))


def test_rejects_arm_jump_from_observed_state():
    gate = ActionSafetyGate(ActionSafetyConfig(max_arm_delta_rad=0.05))
    action = np.zeros((1, 16))
    action[0, 8] = 0.051
    with pytest.raises(ActionSafetyError, match="jump"):
        gate.validate(action, np.zeros(16))


def test_rejects_configured_joint_limit():
    config = ActionSafetyConfig(joint_lower=[-0.1] * 14, joint_upper=[0.1] * 14)
    gate = ActionSafetyGate(config)
    action = np.zeros((1, 16))
    action[0, 0] = 0.2
    with pytest.raises(ActionSafetyError, match="outside configured limits"):
        gate.validate(action, np.zeros(16))

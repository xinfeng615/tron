import numpy as np
import pytest

from experiments.robot.tron2 import msgpack_numpy
from experiments.robot.tron2.serve_openvla import OpenVLATron2Policy, image_to_hwc_uint8


def test_msgpack_numpy_round_trip():
    value = {"actions": np.arange(32, dtype=np.float32).reshape(2, 16)}
    result = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(value))
    np.testing.assert_array_equal(result["actions"], value["actions"])


def test_image_accepts_openpi_chw_layout():
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    assert image_to_hwc_uint8(image).shape == (224, 224, 3)


def test_image_rejects_invalid_shape():
    with pytest.raises(ValueError, match="three dimensions"):
        image_to_hwc_uint8(np.zeros((224, 224), dtype=np.uint8))


def test_proprio_uses_checkpoint_q01_q99_normalization():
    policy = OpenVLATron2Policy.__new__(OpenVLATron2Policy)
    policy.expected_action_dim = 16
    policy.unnorm_key = "tron2_lerobot"
    policy.vla = type(
        "FakeVLA",
        (),
        {
            "norm_stats": {
                "tron2_lerobot": {
                    "proprio": {
                        "q01": np.zeros(16, dtype=np.float32),
                        "q99": np.full(16, 2.0, dtype=np.float32),
                    }
                }
            }
        },
    )()

    normalized = policy._normalize_proprio(np.ones(16, dtype=np.float32))
    np.testing.assert_allclose(normalized, np.zeros(16, dtype=np.float32))


def test_proprio_rejects_wrong_state_dimension():
    policy = OpenVLATron2Policy.__new__(OpenVLATron2Policy)
    policy.expected_action_dim = 16
    policy.unnorm_key = "tron2_lerobot"
    policy.vla = type("FakeVLA", (), {"norm_stats": {}})()
    with pytest.raises(ValueError, match="shape"):
        policy._normalize_proprio(np.zeros(18, dtype=np.float32))

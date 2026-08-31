from pathlib import Path
import sys
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deploy_config import OPENPI_CAMERA_NAMES, format_openvla_obs, infer_with_timing, validate_observation_freshness
from openpi_client import msgpack_numpy


def test_openvla_observation_preserves_openpi_keys_and_state():
    state = np.arange(16, dtype=np.float32)
    obs = {
        "state": state,
        "images": {
            name: np.zeros((480, 640, 3), dtype=np.uint8)
            for name in OPENPI_CAMERA_NAMES
        },
        "metadata": {"not": "sent"},
    }

    formatted = format_openvla_obs(obs, prompt="pick up the banana")

    assert list(formatted["images"]) == OPENPI_CAMERA_NAMES
    assert all(image.shape == (3, 224, 224) for image in formatted["images"].values())
    np.testing.assert_array_equal(formatted["state"], state)
    assert formatted["prompt"] == "pick up the banana"
    assert "metadata" not in formatted


def test_rejects_stale_or_repeated_observation():
    stale = {"metadata": {"observation_ref_timestamp_ms": time.time() * 1000.0 - 1000.0}}
    with pytest.raises(RuntimeError, match="stale"):
        validate_observation_freshness(stale, previous_timestamp_ms=None, max_age_ms=500.0)

    timestamp_ms = time.time() * 1000.0
    repeated = {"metadata": {"observation_ref_timestamp_ms": timestamp_ms}}
    with pytest.raises(RuntimeError, match="did not advance"):
        validate_observation_freshness(
            repeated,
            previous_timestamp_ms=timestamp_ms,
            max_age_ms=500.0,
        )


def test_inference_timeout_is_forwarded_to_websocket():
    class FakeWebsocket:
        def __init__(self):
            self.timeout = None

        def send(self, _data):
            pass

        def recv(self, timeout=None):
            self.timeout = timeout
            return msgpack_numpy.Packer().pack({"actions": np.zeros((1, 16), dtype=np.float32)})

    policy = type("FakePolicy", (), {})()
    policy._packer = msgpack_numpy.Packer()
    policy._ws = FakeWebsocket()

    answer, _ = infer_with_timing(policy, {"state": np.zeros(16)}, timeout_s=1.25)

    assert policy._ws.timeout == 1.25
    assert answer["actions"].shape == (1, 16)

"""Run an OpenVLA policy on TRON2 through the OpenPI-compatible websocket protocol."""

from __future__ import annotations

import argparse
import time

from _external_tron2_env import ensure_external_tron2_env_on_path
from action_safety import ActionSafetyConfig, ActionSafetyError, ActionSafetyGate
from deploy_config import PromptController
from deploy_config import build_env_config
from deploy_config import OPENPI_CAMERA_NAMES
from deploy_config import format_openvla_obs
from deploy_config import infer_with_timing
from deploy_config import load_deploy_config
from deploy_config import policy_host
from deploy_config import policy_port
from deploy_config import positive_int_or_none
from deploy_config import section
from deploy_config import validate_observation_freshness
import numpy as np
from openpi_client import websocket_client_policy

ensure_external_tron2_env_on_path()

from tron2_env import Tron2Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Path to the TRON2 client YAML profile")
    parser.add_argument("--prompt", help="Override client.prompt from the YAML profile")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the robot. Without this flag the client only logs predicted actions.",
    )
    return parser.parse_args()


def validate_server_metadata(metadata: dict) -> None:
    if metadata.get("model_family") != "openvla-m6":
        raise RuntimeError(f"Expected an OpenVLA-M6 server, got metadata={metadata}")
    if int(metadata.get("action_dim", -1)) != 16:
        raise RuntimeError(f"TRON2 client requires a 16-dimensional policy, got metadata={metadata}")
    if int(metadata.get("state_dim", -1)) != 16:
        raise RuntimeError(f"TRON2 client requires a 16-dimensional state input, got metadata={metadata}")
    if metadata.get("camera_names") != OPENPI_CAMERA_NAMES:
        raise RuntimeError(
            f"TRON2 client requires cameras {OPENPI_CAMERA_NAMES} in this order, got metadata={metadata}"
        )
    if metadata.get("rtc_enabled"):
        raise RuntimeError("Use the synchronous OpenVLA server with rtc_enabled=false")


def main() -> None:
    args = parse_args()
    profile = load_deploy_config(args.profile)
    client_profile = section(profile, "client")
    safety_profile = section(profile, "safety")
    env_config = build_env_config(profile)
    if env_config.state_dim != 16:
        raise ValueError(f"OpenVLA TRON2 deployment currently requires client.state_dim=16, got {env_config.state_dim}")

    max_inferences = positive_int_or_none(client_profile.get("max_inferences", 20), field_name="client.max_inferences")
    execution_horizon = int(client_profile.get("execution_horizon", 1))
    if execution_horizon < 1:
        raise ValueError("client.execution_horizon must be positive")
    max_inference_ms = float(safety_profile.get("max_inference_ms", 5000.0))
    max_observation_age_ms = float(safety_profile.get("max_observation_age_ms", 500.0))
    if max_inference_ms <= 0 or max_observation_age_ms <= 0:
        raise ValueError("safety.max_inference_ms and safety.max_observation_age_ms must be positive")
    safety_gate = ActionSafetyGate(ActionSafetyConfig.from_mapping(safety_profile))
    if args.execute and not safety_gate.has_joint_limits:
        raise ValueError(
            "Real execution requires safety.joint_lower and safety.joint_upper with 14 "
            "hardware-approved arm-joint limits"
        )
    prompt = PromptController(args.prompt or client_profile.get("prompt"))
    prompt.start_stdin_listener()

    mode = "EXECUTE" if args.execute else "SHADOW (no env.step calls)"
    print(f"mode: {mode}")
    print(f"observation_source: {env_config.observation_source}")
    print(f"policy: {policy_host(client_profile)}:{policy_port(client_profile)}")

    with Tron2Env(env_config) as env:
        if args.execute:
            print("WARNING: --execute is active; the robot will move after safety validation.")
            env.reset()

        policy = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host(client_profile),
            port=policy_port(client_profile),
        )
        validate_server_metadata(policy.get_server_metadata())

        inference_index = 0
        previous_observation_timestamp_ms = None
        while max_inferences is None or inference_index < max_inferences:
            obs = env.get_obs()
            previous_observation_timestamp_ms = validate_observation_freshness(
                obs,
                previous_timestamp_ms=previous_observation_timestamp_ms,
                max_age_ms=max_observation_age_ms,
            )
            answer, timing = infer_with_timing(
                policy,
                format_openvla_obs(obs, prompt=prompt.get()),
                timeout_s=max_inference_ms / 1000.0,
            )
            if timing["total_ms"] > max_inference_ms:
                raise ActionSafetyError(
                    f"Inference took {timing['total_ms']:.1f} ms; limit is {max_inference_ms:.1f} ms"
                )
            actions = safety_gate.validate(answer["actions"], obs["state"])
            actions = actions[: min(execution_horizon, len(actions))]
            print(
                f"infer={inference_index} H={len(actions)} ws={timing['total_ms']:.1f}ms "
                f"first={np.array2string(actions[0], precision=4, suppress_small=True)}"
            )

            if args.execute:
                for action in actions:
                    started = time.perf_counter()
                    env.step(action)
                    remaining = 1.0 / env_config.fps - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            inference_index += 1


if __name__ == "__main__":
    main()

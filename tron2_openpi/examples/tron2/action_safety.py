"""Client-side validation for TRON2 policy actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

TRON2_ACTION_DIM = 16
TRON2_ARM_INDICES = np.array([*range(0, 7), *range(8, 15)], dtype=np.int64)
TRON2_GRIPPER_INDICES = np.array([7, 15], dtype=np.int64)


class ActionSafetyError(RuntimeError):
    """Raised when a policy action must not be sent to the robot."""


@dataclass(frozen=True)
class ActionSafetyConfig:
    max_arm_delta_rad: float = 0.08
    max_chunk_size: int = 10
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    joint_lower: Optional[Sequence[float]] = None
    joint_upper: Optional[Sequence[float]] = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ActionSafetyConfig":
        return cls(
            max_arm_delta_rad=float(values.get("max_arm_delta_rad", 0.08)),
            max_chunk_size=int(values.get("max_chunk_size", 10)),
            gripper_min=float(values.get("gripper_min", 0.0)),
            gripper_max=float(values.get("gripper_max", 1.0)),
            joint_lower=values.get("joint_lower"),
            joint_upper=values.get("joint_upper"),
        )


class ActionSafetyGate:
    def __init__(self, config: ActionSafetyConfig) -> None:
        self.config = config
        if config.max_arm_delta_rad <= 0:
            raise ValueError("safety.max_arm_delta_rad must be positive")
        if config.max_chunk_size < 1:
            raise ValueError("safety.max_chunk_size must be positive")
        if config.gripper_min >= config.gripper_max:
            raise ValueError("safety.gripper_min must be less than safety.gripper_max")
        self._joint_lower = self._joint_limit(config.joint_lower, "joint_lower")
        self._joint_upper = self._joint_limit(config.joint_upper, "joint_upper")
        if (self._joint_lower is None) != (self._joint_upper is None):
            raise ValueError("safety.joint_lower and safety.joint_upper must be provided together")
        if self._joint_lower is not None and np.any(self._joint_lower >= self._joint_upper):
            raise ValueError("Every safety.joint_lower value must be below joint_upper")

    @property
    def has_joint_limits(self) -> bool:
        return self._joint_lower is not None

    @staticmethod
    def _joint_limit(value: Optional[Sequence[float]], name: str) -> Optional[np.ndarray]:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (len(TRON2_ARM_INDICES),):
            raise ValueError(f"safety.{name} must contain 14 arm-joint values, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"safety.{name} contains NaN or infinity")
        return array

    def validate(self, actions: Any, current_state: Any) -> np.ndarray:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[1] != TRON2_ACTION_DIM:
            raise ActionSafetyError(f"Expected actions [H, 16], got {chunk.shape}")
        if chunk.shape[0] > self.config.max_chunk_size:
            raise ActionSafetyError(
                f"Action chunk H={chunk.shape[0]} exceeds safety.max_chunk_size={self.config.max_chunk_size}"
            )
        if not np.isfinite(chunk).all():
            raise ActionSafetyError("Action chunk contains NaN or infinity")

        state = np.asarray(current_state, dtype=np.float32)
        if state.ndim != 1 or state.size < TRON2_ACTION_DIM or not np.isfinite(state[:TRON2_ACTION_DIM]).all():
            raise ActionSafetyError(f"Current state must contain at least 16 finite values, got {state.shape}")

        result = chunk.copy()
        result[:, TRON2_GRIPPER_INDICES] = np.clip(
            result[:, TRON2_GRIPPER_INDICES], self.config.gripper_min, self.config.gripper_max
        )
        arm_actions = result[:, TRON2_ARM_INDICES]
        if self._joint_lower is not None:
            outside = (arm_actions < self._joint_lower[None]) | (arm_actions > self._joint_upper[None])
            if outside.any():
                frame, joint = np.argwhere(outside)[0]
                raise ActionSafetyError(
                    f"Arm joint {joint} at action frame {frame} is outside configured limits: "
                    f"{arm_actions[frame, joint]:.5f}"
                )

        reference = np.concatenate((state[:7], state[8:15]))[None]
        arm_sequence = np.concatenate((reference, arm_actions), axis=0)
        deltas = np.abs(np.diff(arm_sequence, axis=0))
        if float(np.max(deltas)) > self.config.max_arm_delta_rad:
            frame, joint = np.unravel_index(int(np.argmax(deltas)), deltas.shape)
            raise ActionSafetyError(
                f"Arm joint {joint} jump before action frame {frame} is {deltas[frame, joint]:.5f} rad; "
                f"limit is {self.config.max_arm_delta_rad:.5f} rad"
            )
        return result

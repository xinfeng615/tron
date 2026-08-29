"""Serve an OpenVLA-M6 policy using the TRON2/OpenPI websocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from experiments.robot.tron2 import msgpack_numpy

LOGGER = logging.getLogger("openvla.tron2.server")
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
DEFAULT_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def get_openvla_prompt(instruction: str, model_path: str) -> str:
    instruction = instruction.strip().lower()
    if "v01" in model_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction}?\nOut:"


def image_to_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Camera image must have three dimensions, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Camera image must have three RGB channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Camera image contains NaN or infinity")
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


class OpenVLATron2Policy:
    def __init__(
        self,
        model_path: str,
        *,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        default_prompt: Optional[str] = None,
        unnorm_key: Optional[str] = None,
        expected_action_dim: int = 16,
        attn_implementation: Optional[str] = "flash_attention_2",
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self._torch = torch
        self.model_path = model_path
        self.camera_names: Tuple[str, ...] = tuple(camera_names)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError(f"camera_names must be non-empty and unique, got {self.camera_names}")
        self.default_prompt = default_prompt
        self.unnorm_key = unnorm_key
        self.expected_action_dim = expected_action_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # The classes registered above contain the TRON2 multi-camera/proprio extensions.
        # Do not let stale Python files embedded in a checkpoint override those classes.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        if attn_implementation and self.device.type == "cuda":
            load_kwargs["attn_implementation"] = attn_implementation
        self.vla = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).to(self.device)
        self.vla.eval()

        stats_path = Path(model_path) / "dataset_statistics.json"
        if os.path.isdir(model_path) and stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as stream:
                self.vla.norm_stats = json.load(stream)

        model_action_dim = int(self.vla.get_action_dim(self.unnorm_key))
        if model_action_dim != self.expected_action_dim:
            raise ValueError(
                f"Checkpoint produces {model_action_dim} actions, but TRON2 deployment expects "
                f"{self.expected_action_dim}. Train/export the checkpoint with action_dim=16."
            )
        model_num_views = int(getattr(self.vla.config, "num_image_views", 1))
        if model_num_views != len(self.camera_names):
            raise ValueError(
                f"Checkpoint expects {model_num_views} image views, but deployment provides "
                f"{len(self.camera_names)} cameras: {self.camera_names}"
            )
        if not getattr(self.vla.config, "proprio_enabled", False):
            raise ValueError("TRON2 checkpoint must be trained with proprio_enabled=true")
        model_proprio_dim = int(getattr(self.vla.config, "proprio_dim", 0) or 0)
        if model_proprio_dim != self.expected_action_dim:
            raise ValueError(
                f"Checkpoint expects {model_proprio_dim} proprio values, but TRON2 provides "
                f"{self.expected_action_dim}"
            )

    def _get_dataset_stats(self) -> Dict[str, Any]:
        norm_stats = self.vla.norm_stats
        if self.unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    f"Checkpoint has multiple normalization keys {sorted(norm_stats)}; pass --unnorm-key"
                )
            key = next(iter(norm_stats))
        else:
            key = self.unnorm_key
        if key not in norm_stats:
            raise KeyError(f"Normalization key {key!r} is not in checkpoint statistics: {sorted(norm_stats)}")
        return norm_stats[key]

    def _normalize_proprio(self, state: Any) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.expected_action_dim,):
            raise ValueError(f"TRON2 state must have shape ({self.expected_action_dim},), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("TRON2 state contains NaN or infinity")
        proprio_stats = self._get_dataset_stats().get("proprio")
        if not isinstance(proprio_stats, dict):
            raise KeyError("Checkpoint statistics do not contain proprio normalization values")
        low = np.asarray(proprio_stats["q01"], dtype=np.float32)
        high = np.asarray(proprio_stats["q99"], dtype=np.float32)
        mask = np.asarray(proprio_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        if low.shape != state.shape or high.shape != state.shape or mask.shape != state.shape:
            raise ValueError("Checkpoint proprio statistics do not match the 16-dimensional TRON2 state")
        normalized = np.where(
            mask,
            np.clip(2.0 * (state - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
            state,
        )
        if "min" in proprio_stats and "max" in proprio_stats:
            minimum = np.asarray(proprio_stats["min"], dtype=np.float32)
            maximum = np.asarray(proprio_stats["max"], dtype=np.float32)
            if minimum.shape != state.shape or maximum.shape != state.shape:
                raise ValueError("Checkpoint proprio min/max statistics do not match the TRON2 state")
            normalized = np.where(minimum == maximum, 0.0, normalized)
        return normalized.astype(np.float32)

    @property
    def action_horizon(self) -> int:
        if getattr(self.vla.config, "action_chunk_head_enabled", False):
            return int(getattr(self.vla.config, "action_chunk_size", 1))
        return 1

    def warmup(self) -> None:
        proprio_stats = self._get_dataset_stats()["proprio"]
        state = 0.5 * (
            np.asarray(proprio_stats["q01"], dtype=np.float32)
            + np.asarray(proprio_stats["q99"], dtype=np.float32)
        )
        observation = {
            "images": {
                name: np.zeros((224, 224, 3), dtype=np.uint8)
                for name in self.camera_names
            },
            "state": state,
            "prompt": self.default_prompt or "do something",
        }
        result = self.infer(observation)
        LOGGER.info(
            "Warmup complete: horizon=%d infer_ms=%.1f",
            result["actions"].shape[0],
            result["policy_timing"]["infer_ms"],
        )

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        images = observation.get("images")
        missing_cameras = [name for name in self.camera_names if not isinstance(images, dict) or name not in images]
        if missing_cameras:
            available = sorted(images) if isinstance(images, dict) else []
            raise KeyError(f"Missing cameras {missing_cameras}; available cameras: {available}")
        instruction = observation.get("prompt") or self.default_prompt
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation must contain a non-empty 'prompt'")

        camera_images = [Image.fromarray(image_to_hwc_uint8(images[name])) for name in self.camera_names]
        prompt = get_openvla_prompt(instruction, self.model_path)
        text_inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        pixel_values = self._torch.stack(
            [self.processor.image_processor.apply_transform(image.convert("RGB")) for image in camera_images]
        ).unsqueeze(0)
        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": pixel_values.to(self.device, dtype=self.dtype),
            "proprio": self._torch.from_numpy(self._normalize_proprio(observation.get("state")))
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype),
        }

        started = time.perf_counter()
        with self._torch.inference_mode():
            if self.action_horizon > 1 and hasattr(self.vla, "predict_action_chunk"):
                actions = self.vla.predict_action_chunk(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
            else:
                actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
        infer_ms = (time.perf_counter() - started) * 1000.0

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.expected_action_dim:
            raise ValueError(
                f"Policy output must have shape [H, {self.expected_action_dim}], got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Policy output contains NaN or infinity")
        return {"actions": actions, "policy_timing": {"infer_ms": infer_ms}}


class WebsocketPolicyServer:
    def __init__(self, policy: OpenVLATron2Policy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = {
            "model_family": "openvla-m6",
            "action_dim": policy.expected_action_dim,
            "action_horizon": policy.action_horizon,
            "camera_names": list(policy.camera_names),
            "state_dim": policy.expected_action_dim,
            "rtc_enabled": False,
        }

    async def _handler(self, websocket) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        LOGGER.info("Client connected: %s", websocket.remote_address)
        try:
            async for message in websocket:
                request = msgpack_numpy.unpackb(message)
                if isinstance(request, dict) and request.get("__rtc_request"):
                    raise ValueError("OpenVLA TRON2 server does not support Pi0 RTC requests")
                started = time.perf_counter()
                result = self.policy.infer(request)
                result["server_timing"] = {"infer_ms": (time.perf_counter() - started) * 1000.0}
                await websocket.send(packer.pack(result))
        except Exception:
            error = traceback.format_exc()
            LOGGER.exception("Policy request failed")
            try:
                await websocket.send(error)
                await websocket.close(code=1011, reason="Policy inference failed")
            except Exception:
                pass

    async def run(self) -> None:
        from websockets.asyncio.server import serve

        async with serve(self._handler, self.host, self.port, compression=None, max_size=None) as server:
            LOGGER.info("Serving OpenVLA TRON2 policy at ws://%s:%d", self.host, self.port)
            await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_CAMERA_NAMES))
    parser.add_argument("--default-prompt")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--expected-action-dim", type=int, default=16)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    policy = OpenVLATron2Policy(
        args.model_path,
        camera_names=args.camera_names,
        default_prompt=args.default_prompt,
        unnorm_key=args.unnorm_key,
        expected_action_dim=args.expected_action_dim,
        attn_implementation=args.attn_implementation or None,
    )
    if not args.skip_warmup:
        policy.warmup()
    asyncio.run(WebsocketPolicyServer(policy, args.host, args.port).run())


if __name__ == "__main__":
    main()

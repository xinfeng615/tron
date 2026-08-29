"""Convert a TRON2 LeRobot v3 recording into an OpenVLA-compatible RLDS dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Tuple

try:
    import av
except ImportError as exc:
    raise ImportError("PyAV is required to decode the TRON2 AV1 videos. Install it with: pip install av") from exc
import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET_ENV = "TRON2_LEROBOT_DATASET_PATH"
ACTION_DIM = 16
SOURCE_ACTION_DIM = 18
IMAGE_SHAPE = (480, 640, 3)
CAMERA_FEATURES = {
    "image_primary": "observation.images.cam_high",
    "image_left_wrist": "observation.images.cam_left_wrist",
    "image_right_wrist": "observation.images.cam_right_wrist",
}
LEARNED_JOINT_NAMES = [
    "abad_L_Joint",
    "hip_L_Joint",
    "yaw_L_Joint",
    "knee_L_Joint",
    "wrist_yaw_L_Joint",
    "wrist_pitch_L_Joint",
    "wrist_roll_L_Joint",
    "left_gripper",
    "abad_R_Joint",
    "hip_R_Joint",
    "yaw_R_Joint",
    "knee_R_Joint",
    "wrist_yaw_R_Joint",
    "wrist_pitch_R_Joint",
    "wrist_roll_R_Joint",
    "right_gripper",
]
SOURCE_HEAD_JOINT_NAMES = ["head_pitch_Joint", "head_yaw_Joint"]
PARQUET_COLUMNS = ["episode_index", "frame_index", "task_index", "observation.state", "action"]


class LocalOnlyDatasetInfo(tfds.core.DatasetInfo):
    """Metadata for a private dataset that must not query the public TFDS bucket."""

    def initialize_from_bucket(self) -> None:
        return None


def _as_learned_vector(value: Any, *, name: str, episode_index: int, frame_index: int) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (SOURCE_ACTION_DIM,):
        raise ValueError(
            f"Episode {episode_index}, frame {frame_index}: {name} has shape {vector.shape}; "
            f"expected ({SOURCE_ACTION_DIM},)"
        )
    vector = vector[:ACTION_DIM].copy()
    if not np.isfinite(vector).all():
        raise ValueError(f"Episode {episode_index}, frame {frame_index}: {name} contains NaN/Inf")
    return vector


def _load_and_validate_metadata(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root; missing {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot codebase_version 'v3.0', got {info.get('codebase_version')!r}")
    if int(info.get("fps", -1)) != 30:
        raise ValueError(f"Expected 30 Hz TRON2 data, got fps={info.get('fps')!r}")

    features = info.get("features", {})
    for source_key in CAMERA_FEATURES.values():
        feature = features.get(source_key, {})
        if tuple(feature.get("shape", ())) != IMAGE_SHAPE:
            raise ValueError(f"Camera {source_key!r} is missing or has unexpected shape {feature.get('shape')!r}")

    expected_joint_names = LEARNED_JOINT_NAMES + SOURCE_HEAD_JOINT_NAMES
    for source_key in ("observation.state", "action"):
        feature = features.get(source_key, {})
        if tuple(feature.get("shape", ())) != (SOURCE_ACTION_DIM,):
            raise ValueError(f"Feature {source_key!r} must have shape ({SOURCE_ACTION_DIM},)")
        if list(feature.get("names") or []) != expected_joint_names:
            raise ValueError(f"Feature {source_key!r} has an unexpected joint order")

    if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("LeRobot metadata reports an empty dataset")
    return info


def _load_tasks(root: Path) -> Dict[int, str]:
    task_path = root / "meta" / "tasks.parquet"
    if not task_path.is_file():
        raise FileNotFoundError(f"Missing task metadata: {task_path}")
    values = pq.read_table(task_path).to_pydict()
    task_indices = values.get("task_index")
    texts = values.get("task") or values.get("instruction") or values.get("__index_level_0__")
    if task_indices is None or texts is None:
        raise ValueError(f"Unsupported tasks.parquet columns: {sorted(values)}")
    tasks = {int(index): str(text).strip() for index, text in zip(task_indices, texts)}
    if not tasks or any(not text for text in tasks.values()):
        raise ValueError("tasks.parquet contains an empty language instruction")
    return tasks


def _open_videos(paths: Mapping[str, Path]) -> Dict[str, Tuple[Any, Iterator[Any]]]:
    readers: Dict[str, Tuple[Any, Iterator[Any]]] = {}
    try:
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing TRON2 video: {path}")
            container = av.open(str(path), mode="r")
            if not container.streams.video:
                container.close()
                raise ValueError(f"Video has no decodable stream: {path}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            readers[name] = (container, iter(container.decode(stream)))
    except Exception:
        for container, _ in readers.values():
            container.close()
        raise
    return readers


def _read_rgb_frame(reader: Tuple[Any, Iterator[Any]], path: Path, frame_index: int) -> np.ndarray:
    _, frames = reader
    try:
        video_frame = next(frames)
    except StopIteration as exc:
        raise ValueError(f"Video ended before parquet frame {frame_index}: {path}") from exc
    frame = video_frame.to_ndarray(format="rgb24")
    if frame.shape != IMAGE_SHAPE:
        raise ValueError(f"Unexpected frame shape in {path}: {frame.shape}; expected {IMAGE_SHAPE}")
    return frame


def _episode_steps(
    table: Any,
    video_paths: Mapping[str, Path],
    episode_index: int,
    instruction: str,
) -> Iterator[dict]:
    row_count = len(table)
    readers = _open_videos(video_paths)
    try:
        for row_index in range(row_count):
            if row_index % 250 == 0:
                logging.info("Episode %d: encoding frame %d/%d", episode_index, row_index, row_count)
            images = {
                name: _read_rgb_frame(readers[name], path, row_index)
                for name, path in video_paths.items()
            }
            state = _as_learned_vector(
                table["observation.state"][row_index],
                name="observation.state",
                episode_index=episode_index,
                frame_index=row_index,
            )
            action = _as_learned_vector(
                table["action"][row_index],
                name="action",
                episode_index=episode_index,
                frame_index=row_index,
            )
            is_last = row_index == row_count - 1
            yield {
                "observation": {**images, "state": state},
                "action": action,
                "discount": np.float32(1.0),
                "reward": np.float32(is_last),
                "is_first": row_index == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "language_instruction": instruction,
            }

        for name, (_, frames) in readers.items():
            try:
                next(frames)
            except StopIteration:
                continue
            raise ValueError(f"Video contains more frames than its parquet file: {video_paths[name]}")
    finally:
        for container, _ in readers.values():
            container.close()


class Tron2Lerobot(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Store three RGB views as JPEG and validate LeRobot v3 metadata, joints, and frame indices."
    }

    def _info(self):
        def image_feature():
            return tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg")

        return LocalOnlyDatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_primary": image_feature(),
                                    "image_left_wrist": image_feature(),
                                    "image_right_wrist": image_feature(),
                                    "state": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_index": tfds.features.Scalar(dtype=np.int64),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        root = os.environ.get(DATASET_ENV)
        if not root:
            raise ValueError(f"Set {DATASET_ENV} to the LeRobot dataset directory")
        root_path = Path(root)
        metadata = _load_and_validate_metadata(root_path)
        return {"train": self._generate_examples(root_path, metadata)}

    def _generate_examples(self, root: Path, metadata: dict) -> Iterator[Tuple[str, dict]]:
        tasks = _load_tasks(root)
        data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        expected_episode_count = int(metadata["total_episodes"])
        if len(data_files) != expected_episode_count:
            raise ValueError(
                f"Found {len(data_files)} parquet files, but info.json reports {expected_episode_count} episodes"
            )

        seen_episode_indices = set()
        total_rows = 0
        for data_path in data_files:
            table = pq.read_table(data_path, columns=PARQUET_COLUMNS)
            row_count = len(table)
            if row_count == 0:
                raise ValueError(f"Empty episode file: {data_path}")

            episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
            if not np.all(episode_values == episode_values[0]):
                raise ValueError(f"Parquet file contains more than one episode: {data_path}")
            episode_index = int(episode_values[0])
            if episode_index in seen_episode_indices:
                raise ValueError(f"Duplicate episode_index {episode_index}")
            seen_episode_indices.add(episode_index)

            frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
            if not np.array_equal(frame_indices, np.arange(row_count, dtype=np.int64)):
                raise ValueError(f"Episode {episode_index} has non-contiguous frame_index values")

            task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64).reshape(-1)
            if not np.all(task_values == task_values[0]):
                raise ValueError(f"Episode {episode_index} contains more than one task_index")
            task_index = int(task_values[0])
            if task_index not in tasks:
                raise KeyError(f"Task index {task_index} is missing from meta/tasks.parquet")

            relative_video_path = data_path.relative_to(root / "data").with_suffix(".mp4")
            video_paths = {
                feature_name: root / "videos" / camera_name / relative_video_path
                for feature_name, camera_name in CAMERA_FEATURES.items()
            }
            logging.info(
                "Preparing episode %d with %d frames and instruction %r",
                episode_index,
                row_count,
                tasks[task_index],
            )
            total_rows += row_count
            yield str(episode_index), {
                "steps": _episode_steps(table, video_paths, episode_index, tasks[task_index]),
                "episode_metadata": {"file_path": str(data_path), "episode_index": episode_index},
            }

        expected_episode_indices = set(range(expected_episode_count))
        if seen_episode_indices != expected_episode_indices:
            raise ValueError(
                f"Episode IDs do not match info.json: found {sorted(seen_episode_indices)}, "
                f"expected {sorted(expected_episode_indices)}"
            )
        expected_frame_count = int(metadata["total_frames"])
        if total_rows != expected_frame_count:
            raise ValueError(f"Found {total_rows} parquet rows, but info.json reports {expected_frame_count} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    os.environ[DATASET_ENV] = str(args.dataset_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = Tron2Lerobot(data_dir=str(args.output_dir.resolve()))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False),
        file_format="tfrecord",
    )
    logging.info("Prepared %s at %s", builder.info.full_name, builder.data_dir)


if __name__ == "__main__":
    main()

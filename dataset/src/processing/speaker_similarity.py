"""Build per-video face embeddings and pairwise similarities without clustering."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.processing.clean_videos import (
    DEFAULT_CONFIG,
    CleaningConfig,
    _gender_value,
    _normalized_embedding,
    load_config,
)
from src.processing.speaker_identity_debug import (
    TIMESTAMPS_SEC,
    SpeakerIdentityDebugError,
    _UPPER_FACE_CENTER_RATIO,
    _create_face_model,
    _publish,
    _select_presenter_face,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "videos"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "speaker_similarity_v1"
)
TOP_K = 5


class SpeakerSimilarityError(RuntimeError):
    """Raised when speaker similarity artifacts cannot be built safely."""


def _gender(face: Any) -> str | None:
    try:
        return _gender_value(face)
    except (TypeError, ValueError, OverflowError):
        return None


def _aggregate_gender(values: list[str]) -> str:
    counts = Counter(values)
    if counts["male"] > counts["female"]:
        return "male"
    if counts["female"] > counts["male"]:
        return "female"
    return "unknown"


def _video_embedding(
    video: Path, face_model: Any
) -> tuple[np.ndarray | None, int, str, list[dict[str, Any]]]:
    """Average selected normalized face embeddings and normalize the result."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None, 0, "unknown", [
            {
                "timestamp_sec": timestamp,
                "embedding_available": False,
                "gender": None,
                "error": "video_open_failed",
            }
            for timestamp in TIMESTAMPS_SEC
        ]

    embeddings: list[np.ndarray] = []
    genders: list[str] = []
    frame_results: list[dict[str, Any]] = []
    try:
        for timestamp in TIMESTAMPS_SEC:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                frame_results.append(
                    {
                        "timestamp_sec": timestamp,
                        "embedding_available": False,
                        "gender": None,
                        "error": "frame_decode_failed",
                    }
                )
                continue

            faces = list(face_model.get(frame))
            selected, selection_scope = _select_presenter_face(faces, frame.shape[0])
            if selected is None:
                frame_results.append(
                    {
                        "timestamp_sec": timestamp,
                        "embedding_available": False,
                        "gender": None,
                        "error": "no_valid_face",
                    }
                )
                continue

            embedding = _normalized_embedding(selected)
            gender = _gender(selected)
            frame_results.append(
                {
                    "timestamp_sec": timestamp,
                    "embedding_available": embedding is not None,
                    "gender": gender,
                    "selection_scope": selection_scope,
                    "error": None if embedding is not None else "embedding_unavailable",
                }
            )
            if embedding is None:
                continue
            if embeddings and embedding.shape != embeddings[0].shape:
                frame_results[-1]["embedding_available"] = False
                frame_results[-1]["error"] = "embedding_shape_mismatch"
                continue
            embeddings.append(embedding.astype(np.float32, copy=False))
            if gender is not None:
                genders.append(gender)
    finally:
        capture.release()

    if not embeddings:
        return None, 0, _aggregate_gender(genders), frame_results
    mean = np.mean(np.stack(embeddings, axis=0), axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0.0 or not np.isfinite(mean).all():
        return None, 0, _aggregate_gender(genders), frame_results
    return mean / norm, len(embeddings), _aggregate_gender(genders), frame_results


def _similarity_matrix(embeddings: np.ndarray, valid: np.ndarray) -> np.ndarray:
    count = len(embeddings)
    matrix = np.full((count, count), np.nan, dtype=np.float32)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return matrix
    values = embeddings[indices]
    similarities = np.clip(values @ values.T, -1.0, 1.0).astype(np.float32)
    similarities[np.diag_indices_from(similarities)] = 1.0
    matrix[np.ix_(indices, indices)] = similarities
    return matrix


def _summary(matrix: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return {
            "non_diagonal_similarity": None,
            "per_video_top1_similarity": None,
        }

    valid_matrix = matrix[np.ix_(indices, indices)].astype(np.float64)
    unique_pairs = valid_matrix[np.triu_indices(len(indices), k=1)]
    top1 = np.max(
        np.where(np.eye(len(indices), dtype=bool), -np.inf, valid_matrix), axis=1
    )

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "min": round(float(np.min(values)), 6),
            "mean": round(float(np.mean(values)), 6),
            "median": round(float(np.median(values)), 6),
            "max": round(float(np.max(values)), 6),
        }

    return {
        "non_diagonal_similarity": {
            "definition": "unique unordered valid-video pairs (upper triangle)",
            "pair_count": int(len(unique_pairs)),
            **stats(unique_pairs),
        },
        "per_video_top1_similarity": {
            "video_count": int(len(top1)),
            **stats(top1),
        },
    }


def _neighbors(
    videos: list[Path],
    matrix: np.ndarray,
    valid: np.ndarray,
    genders: list[str],
    accepted_frames: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    valid_indices = np.flatnonzero(valid)
    for index, video in enumerate(videos):
        neighbors: list[dict[str, Any]] = []
        if valid[index]:
            candidates = [candidate for candidate in valid_indices if candidate != index]
            candidates.sort(key=lambda candidate: (-float(matrix[index, candidate]), videos[candidate].name))
            neighbors = [
                {
                    "video": videos[candidate].name,
                    "similarity": round(float(matrix[index, candidate]), 6),
                }
                for candidate in candidates[:TOP_K]
            ]
        records.append(
            {
                "video": video.name,
                "status": "valid" if valid[index] else "unresolved",
                "gender": genders[index],
                "accepted_frames": int(accepted_frames[index]),
                "neighbors": neighbors,
            }
        )
    return records


def build_similarity_artifacts(
    input_root: Path,
    output: Path,
    config: CleaningConfig,
    *,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Process all top-level videos and atomically publish similarity artifacts."""
    input_root = input_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not input_root.is_dir():
        raise SpeakerSimilarityError(f"input directory not found: {input_root}")
    videos = sorted(
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() in config.video_extensions
    )
    if not videos:
        raise SpeakerSimilarityError(f"no supported videos found in: {input_root}")
    if output.exists() and not force:
        raise SpeakerSimilarityError(
            f"output already exists (use --force to replace it): {output}"
        )

    face_model, actual_device = _create_face_model(config, device)
    video_embeddings: list[np.ndarray | None] = []
    accepted_counts: list[int] = []
    genders: list[str] = []
    unresolved_reasons: dict[str, list[dict[str, Any]]] = {}
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        embedding, accepted, gender, frames = _video_embedding(video, face_model)
        video_embeddings.append(embedding)
        accepted_counts.append(accepted)
        genders.append(gender)
        if embedding is None:
            unresolved_reasons[video.name] = frames

    valid = np.asarray([embedding is not None for embedding in video_embeddings], dtype=bool)
    embedding_dim = next(
        (int(embedding.shape[0]) for embedding in video_embeddings if embedding is not None),
        0,
    )
    embeddings = np.full((len(videos), embedding_dim), np.nan, dtype=np.float32)
    for index, embedding in enumerate(video_embeddings):
        if embedding is None:
            continue
        if embedding.shape != (embedding_dim,):
            raise SpeakerSimilarityError(
                f"inconsistent embedding shape for {videos[index].name}: {embedding.shape}"
            )
        embeddings[index] = embedding
    accepted_frames = np.asarray(accepted_counts, dtype=np.int32)
    matrix = _similarity_matrix(embeddings, valid)
    statistics = _summary(matrix, valid)
    neighbor_records = _neighbors(
        videos, matrix, valid, genders, accepted_frames
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        np.savez_compressed(
            staging / "video_embeddings.npz",
            videos=np.asarray([video.name for video in videos]),
            embeddings=embeddings,
            valid=valid,
            accepted_frames=accepted_frames,
            gender=np.asarray(genders),
            timestamps_sec=np.asarray(TIMESTAMPS_SEC, dtype=np.float64),
            model=np.asarray(config.insightface_name),
        )
        np.save(staging / "similarity_matrix.npy", matrix, allow_pickle=False)
        with (staging / "top5_neighbors.jsonl").open("w", encoding="utf-8") as target:
            for record in neighbor_records:
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        manifest = {
            "version": "speaker_similarity_v1",
            "complete": True,
            "input_root": str(input_root),
            "video_total": len(videos),
            "valid_videos": int(np.sum(valid)),
            "unresolved_videos": int(np.sum(~valid)),
            "unresolved": sorted(unresolved_reasons),
            "sampling_timestamps_sec": list(TIMESTAMPS_SEC),
            "embedding_model": config.insightface_name,
            "face_detection_size": config.face_detection_size,
            "device": actual_device,
            "face_selection": {
                "prefer_face_center_above_image_ratio": _UPPER_FACE_CENTER_RATIO,
                "rank": "largest_bbox_area",
                "fallback": "largest_bbox_area_in_full_frame",
            },
            "video_embedding": (
                "L2-normalized arithmetic mean of all successfully selected "
                "per-frame normed embeddings"
            ),
            "cosine_similarity": (
                "dot product of L2-normalized video embeddings; unresolved rows and "
                "columns are NaN; valid diagonal is 1"
            ),
            "matrix_order": [video.name for video in videos],
            "embedding_dimension": embedding_dim,
            "statistics": statistics,
            "files": {
                "video_embeddings": "video_embeddings.npz",
                "similarity_matrix": "similarity_matrix.npy",
                "top5_neighbors": "top5_neighbors.jsonl",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output, force)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _format_stats(label: str, values: dict[str, Any] | None) -> str:
    if values is None:
        return f"{label}: n/a"
    return (
        f"{label}: min={values['min']:.6f} mean={values['mean']:.6f} "
        f"median={values['median']:.6f} max={values['max']:.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build video face embeddings and pairwise cosine similarities"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace existing output")
    args = parser.parse_args(argv)
    try:
        manifest = build_similarity_artifacts(
            args.input,
            args.output,
            load_config(args.config.resolve()),
            device=args.device,
            force=args.force,
        )
    except (
        OSError,
        ValueError,
        SpeakerIdentityDebugError,
        SpeakerSimilarityError,
        yaml.YAMLError,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Videos: {manifest['video_total']}")
    print(f"Valid: {manifest['valid_videos']}")
    print(f"Unresolved: {manifest['unresolved_videos']}")
    statistics = manifest["statistics"]
    print(_format_stats("Non-diagonal similarity", statistics["non_diagonal_similarity"]))
    print(_format_stats("Per-video Top-1", statistics["per_video_top1_similarity"]))
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

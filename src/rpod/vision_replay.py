"""Saved-image replay. No renderer, estimator, controller, or engine is invoked."""

import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .vision_records import read_jsonl, read_rgb, sha256_file


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "perception_dataset" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported perception dataset schema")
    observations = read_jsonl(directory / "observations.jsonl")
    if len(observations) != manifest["frame_count"]:
        raise ValueError("dataset frame count disagrees with observation records")
    for row in observations:
        path = (directory / row["image"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("dataset image escapes the dataset directory")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError(f"dataset image checksum mismatch: {row['image']}")
    return manifest, observations


def export_dataset(directory):
    directory = Path(directory)
    manifest, rows = load_dataset(directory)
    truth = {row["capture_id"]: row for row in read_jsonl(directory / "truth" / "poses.jsonl")}
    output = directory / "replay.rrd"
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/camera", name="Saved RGB / detected corners"),
                rrb.Spatial3DView(origin="/pose", name="Target pose in optical camera coordinates (m)", line_grid=False),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/metrics/translation_error_m", name="Translation error (m)"),
                rrb.TimeSeriesView(origin="/metrics/rotation_error_deg", name="Rotation error (deg)"),
                rrb.TextLogView(origin="/status", name="Detection status"),
            ),
        ),
        rrb.TimePanel(timeline="fixture", play_state="paused"),
        collapse_panels=True,
    )
    recording = rr.RecordingStream("rpod-perception-dataset")
    try:
        recording.save(output, default_blueprint=blueprint)
        recording.log("/pose", rr.ViewCoordinates.RDF, static=True)
        for index, row in enumerate(rows):
            recording.set_time("fixture", sequence=index)
            recording.log("/camera/rgb", rr.Image(read_rgb(directory / row["image"])))
            pose = row["pose"]
            label = truth[row["capture_id"]]
            if label.get("target_present", True):
                recording.log(
                    "/pose/truth",
                    rr.Boxes3D(
                        sizes=[2, 2, 2], colors=[70, 170, 255],
                        fill_mode=rr.components.FillMode.MajorWireframe,
                    ),
                    rr.Transform3D(
                        translation=label["translation_ct"], mat3x3=label["rotation_ct"]
                    ),
                )
            else:
                recording.log("/pose/truth", rr.Clear(recursive=True))
            if pose["image_corners"] is not None:
                recording.log("/camera/corners", rr.Points2D(pose["image_corners"], radii=3, colors=[255, 170, 60]))
            else:
                recording.log("/camera/corners", rr.Clear(recursive=True))
            if pose["valid"]:
                recording.log(
                    "/pose/estimate",
                    rr.Boxes3D(
                        sizes=[2, 2, 2], colors=[255, 170, 60],
                        fill_mode=rr.components.FillMode.MajorWireframe,
                    ),
                    rr.Transform3D(
                        translation=pose["translation_ct"], mat3x3=pose["rotation_ct"]
                    ),
                )
                recording.log("/metrics/translation_error_m", rr.Scalars(label["translation_error_m"]))
                recording.log("/metrics/rotation_error_deg", rr.Scalars(label["rotation_error_deg"]))
            else:
                recording.log("/pose/estimate", rr.Clear(recursive=True))
            recording.log("/status", rr.TextLog(f"frame={row['capture_id']} valid={pose['valid']} reason={pose['reason']} ids={pose['marker_ids']}"))
        recording.flush()
    finally:
        recording.disconnect()
    return output


def replay_dataset(directory, save_only=False):
    output = export_dataset(directory)
    if not save_only:
        from .visualization import open_replay

        open_replay(output)
    return output

"""Derived Rerun views of saved numerical trajectories."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .frames import relative_state


TARGET_COLOR = [70, 170, 255]
CHASER_COLOR = [255, 170, 60]
AXIS_COLORS = [[230, 80, 80], [80, 210, 100], [90, 140, 255]]
PHASE_NAMES = ("hold", "approach", "retreat", "complete", "acquiring", "braking", "nav_lost")


def _control_data(manifest: dict, trajectory: dict, basis_nl: np.ndarray) -> dict:
    """Derive M2 visual quantities from recorded states, never from a controller."""
    from .frames import port_relative_state, quaternion_matrices

    rotations_nb = quaternion_matrices(trajectory["quaternions_xyzw"])
    rotations_lb = basis_nl[:, None].swapaxes(-1, -2) @ rotations_nb
    ports_body = np.asarray(manifest["docking_ports_body_m"])
    port_position, port_velocity, _ = port_relative_state(
        trajectory["positions_m"],
        trajectory["velocities_mps"],
        trajectory["quaternions_xyzw"],
        trajectory["angular_velocities_body_radps"],
        ports_body,
    )
    alignment_cosine = (
        np.einsum("tij,tij->t", rotations_nb[:, 0], rotations_nb[:, 1]) - 1
    ) / 2
    return {
        "rotations_lb": rotations_lb,
        "ports_body_m": ports_body,
        "port_offsets_local_m": np.einsum("tbij,bj->tbi", rotations_lb, ports_body),
        "port_gap_m": -port_position[:, 1],
        "port_velocity_mps": port_velocity,
        "alignment_deg": np.degrees(np.arccos(np.clip(alignment_cosine, -1, 1))),
        "desired_gap_m": trajectory["desired_gap_m"],
        "phase": trajectory["phase"],
        "requested_forces_n": np.linalg.norm(trajectory["requested_forces_body_n"], axis=-1),
        "applied_forces_n": np.linalg.norm(trajectory["applied_forces_body_n"], axis=-1),
        "requested_torques_nm": np.linalg.norm(trajectory["requested_torques_body_nm"], axis=-1),
        "applied_torques_nm": np.linalg.norm(trajectory["applied_torques_body_nm"], axis=-1),
    }


def _control_panels(control: dict, ranges: np.ndarray, time_axis: rrb.TimeAxis) -> list:
    gap_values = np.concatenate((ranges, control["port_gap_m"], control["desired_gap_m"]))
    gap_padding = max(0.5, float(np.ptp(gap_values)) * 0.05)
    port_speed = np.linalg.norm(control["port_velocity_mps"], axis=-1)
    speed_maximum = max(0.001, float(port_speed.max())) * 1.05
    alignment_maximum = max(1.0, float(control["alignment_deg"].max())) * 1.05
    force_maximum = max(
        0.01,
        float(control["requested_forces_n"].max(initial=0)),
        float(control["applied_forces_n"].max(initial=0)),
    ) * 1.05
    torque_maximum = max(
        0.001,
        float(control["requested_torques_nm"].max(initial=0)),
        float(control["applied_torques_nm"].max(initial=0)),
    ) * 1.05
    phase_maximum = max(3, int(control["phase"].max(initial=0)))
    return [
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="/telemetry",
                contents=["/telemetry/range_m", "/telemetry/port_gap_m", "/telemetry/desired_gap_m"],
                name="COM range / port gap / desired gap (m)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(
                    range=[float(gap_values.min()) - gap_padding, float(gap_values.max()) + gap_padding]
                ),
            ),
            rrb.TimeSeriesView(
                origin="/telemetry/port_velocity_mps",
                name="Port relative velocity in LVLH (m/s)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(range=[-speed_maximum, speed_maximum]),
            ),
            rrb.TimeSeriesView(
                origin="/telemetry/alignment_deg",
                name="Relative body attitude (deg)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(range=[0, alignment_maximum]),
            ),
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="/telemetry/force_n",
                name="Requested / applied body force norms (N)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(range=[0, force_maximum]),
            ),
            rrb.TimeSeriesView(
                origin="/telemetry/torque_nm",
                name="Requested / applied body torque norms (N m)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(range=[0, torque_maximum]),
            ),
            rrb.Tabs(
                rrb.TimeSeriesView(
                    origin="/telemetry/phase",
                    name=(
                        "Phase: 0 hold, 1 approach, 2 retreat, 3 complete"
                        if phase_maximum == 3 else
                        "Phase: 0 hold, 1 approach, 2 retreat, 3 complete, 4 acquire, 5 brake, 6 lost"
                    ),
                    axis_x=time_axis,
                    axis_y=rrb.ScalarAxis(range=[-0.1, phase_maximum + 0.1]),
                ),
                rrb.TextLogView(origin="/events", name="Events"),
                active_tab=0,
            ),
        ),
    ]


def _blueprint(
    manifest: dict,
    positions_m: np.ndarray,
    local_position: np.ndarray,
    local_velocity: np.ndarray,
    times: np.ndarray,
    control: dict | None = None,
) -> rrb.Blueprint:
    orbit_radius_km = max(
        manifest["earth_radius_m"] / 1000,
        float(np.linalg.norm(positions_m, axis=-1).max()) / 1000,
    )
    # Include the target, the complete relative path, and the frame triad in the fit.
    encounter_points = np.concatenate((local_position, np.zeros((1, 3)), np.eye(3) * 8))
    minimum = encounter_points.min(axis=0) - 2
    maximum = encounter_points.max(axis=0) + 2
    center = (minimum + maximum) / 2
    radius = max(float(np.linalg.norm(maximum - minimum) / 2), 5.0)
    eye_direction = np.array([1.2, 0.4, 0.9])
    eye_direction /= np.linalg.norm(eye_direction)
    start_ns = int(times[0].astype(np.int64))
    end_ns = max(int(times[-1].astype(np.int64)), start_ns + 1)
    time_axis = rrb.TimeAxis(
        link=rrb.components.LinkAxis.Independent,
        view_range=rr.TimeRange(
            start=rrb.TimeRangeBoundary.absolute(nanos=start_ns),
            end=rrb.TimeRangeBoundary.absolute(nanos=end_ns),
        ),
    )
    ranges = np.linalg.norm(local_position, axis=-1)
    range_center = float(ranges.min() + ranges.max()) / 2
    range_half_span = max(0.5, float(ranges.max() - ranges.min()) * 0.55)
    speed_maximum = max(0.001, float(np.linalg.norm(local_velocity, axis=-1).max()))
    position_minimum, position_maximum = float(local_position.min()), float(local_position.max())
    position_padding = max(0.1, (position_maximum - position_minimum) * 0.05)
    telemetry_rows = [
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="/telemetry/range_m", name="Separation (m)", axis_x=time_axis,
                axis_y=rrb.ScalarAxis(
                    range=[range_center - range_half_span, range_center + range_half_span]
                ),
            ),
            rrb.TimeSeriesView(
                origin="/telemetry/position_m", name="Target LVLH position (m)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(
                    range=[position_minimum - position_padding, position_maximum + position_padding]
                ),
            ),
            rrb.TimeSeriesView(
                origin="/telemetry/speed_mps", name="Target LVLH speed (m/s)",
                axis_x=time_axis,
                axis_y=rrb.ScalarAxis(range=[0, speed_maximum * 1.05]),
            ),
        )
    ] if control is None else _control_panels(control, ranges, time_axis)
    if manifest.get("schema_version") == 3:
        telemetry_rows.append(rrb.Horizontal(
            rrb.Spatial2DView(origin="/camera", name="Delivered RGB / detected corners"),
            rrb.TimeSeriesView(
                origin="/vision/age_s", name="Last accepted capture age (s)", axis_x=time_axis,
            ),
            rrb.TimeSeriesView(
                origin="/vision/health", name="Navigation: -1 none, 0 fresh, 1 predicted, 2 lost",
                axis_x=time_axis,
            ),
            rrb.Tabs(
                rrb.TimeSeriesView(
                    origin="/vision/position_error_m", name="Estimated target position error (m)",
                    axis_x=time_axis,
                ),
                rrb.TimeSeriesView(
                    origin="/vision/attitude_error_deg", name="Estimated target attitude error (deg)",
                    axis_x=time_axis,
                ),
                rrb.TextLogView(origin="/vision/status", name="Image capture / delivery status"),
            ),
        ))
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/orbit",
                    name="Earth-centered orbit (km; enlarged markers)",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=[0, 0, 3 * orbit_radius_km],
                        look_target=[0, 0, 0],
                        eye_up=[0, 1, 0],
                    ),
                ),
                rrb.Spatial3DView(
                    origin="/encounter",
                    name=(
                        "Target LVLH encounter (m; prescribed orientation)"
                        if control is None else "Target LVLH encounter (m; recorded attitude)"
                    ),
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=center + eye_direction * (2.6 * radius),
                        look_target=center,
                        eye_up=[0, 0, 1],
                    ),
                ),
            ),
            *telemetry_rows,
            row_shares=[3, *([1] * len(telemetry_rows))],
        ),
        rrb.TimePanel(timeline="sim_time", play_state="paused"),
        collapse_panels=True,
    )


def export_replay(
    manifest: dict,
    trajectory: dict[str, np.ndarray],
    output: Path,
    events: list[dict] | None = None,
    *,
    observations: list[dict] | None = None,
    estimates: list[dict] | None = None,
    image_root: Path | None = None,
    lidar: list[dict] | None = None,
) -> Path:
    """Save a headless replay; all geometry and telemetry derive from the record."""
    output = Path(output)
    positions = trajectory["positions_m"]
    local_position, local_velocity, basis_nl = relative_state(
        positions, trajectory["velocities_mps"]
    )
    ticks = trajectory["tick"]
    dt_ns = manifest["dt_ns"]
    if len(ticks) != len(local_position) or not len(ticks):
        raise ValueError("replay requires matching, nonempty state and tick arrays")
    if int(ticks[-1]) * dt_ns > np.iinfo(np.int64).max:
        raise ValueError("replay exceeds the signed 64-bit nanosecond timeline")
    times = (ticks.astype(np.int64) * dt_ns).astype("timedelta64[ns]")
    indexes = [rr.TimeColumn("sim_time", duration=times)]
    control = (
        _control_data(manifest, trajectory, basis_nl)
        if manifest.get("schema_version", 1) in (2, 3) else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(f"rpod-{manifest['scenario']}")
    try:
        recording.save(
            output,
            default_blueprint=_blueprint(
                manifest, positions, local_position, local_velocity, times, control
            ),
        )
        for root in ("/orbit", "/encounter"):
            recording.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Orbit geometry uses kilometers to keep Earth and its nearby paths readable.
        recording.log(
            "/orbit/earth",
            rr.Ellipsoids3D(
                radii=[manifest["earth_radius_m"] / 1000],
                colors=[[35, 75, 125]],
                fill_mode="solid",
            ),
            static=True,
        )
        for body, color in enumerate((TARGET_COLOR, CHASER_COLOR)):
            name = manifest["body_order"][body]
            path = f"/orbit/{name}"
            recording.log(
                path,
                rr.Points3D.from_fields(
                    radii=35, colors=color, labels=[name], show_labels=True
                ),
                static=True,
            )
            recording.send_columns(
                path,
                indexes=indexes,
                columns=rr.Points3D.columns(positions=positions[:, body] / 1000),
            )
            recording.log(
                f"/orbit/paths/{name}",
                rr.LineStrips3D(
                    positions[:, body] / 1000, colors=color, radii=2
                ),
                static=True,
            )

        # Bodies are expressed in target LVLH; M1 prescribes identity body rotations.
        recording.log(
            "/encounter/target",
            rr.Boxes3D(
                sizes=[2, 2, 2], colors=TARGET_COLOR, labels=["target"],
                show_labels=True, fill_mode="solid",
            ),
            static=True,
        )
        recording.log(
            "/encounter/chaser",
            rr.Boxes3D(
                sizes=[1, 1, 1], colors=CHASER_COLOR, labels=["chaser"],
                show_labels=True, fill_mode="solid",
            ),
            static=True,
        )
        if control is None:
            recording.send_columns(
                "/encounter/chaser",
                indexes=indexes,
                columns=rr.Transform3D.columns(translation=local_position),
            )
        else:
            for body, name in enumerate(manifest["body_order"]):
                recording.send_columns(
                    f"/encounter/{name}",
                    indexes=indexes,
                    columns=rr.Transform3D.columns(
                        translation=local_position if body == 1 else np.zeros_like(local_position),
                        mat3x3=control["rotations_lb"][:, body],
                    ),
                )
                recording.log(
                    f"/encounter/{name}/port",
                    rr.Points3D(
                        [control["ports_body_m"][body]],
                        radii=rr.Radius.ui_points(3),
                        colors=TARGET_COLOR if body == 0 else CHASER_COLOR,
                        labels=[f"{name} docking port"],
                        show_labels=False,
                    ),
                    static=True,
                )
            recording.log(
                "/encounter/target/axes",
                rr.Arrows3D(vectors=np.eye(3) * 2, colors=AXIS_COLORS),
                static=True,
            )
            port_positions = control["port_offsets_local_m"].copy()
            port_positions[:, 1] += local_position
            recording.log(
                "/encounter/port_gap_line",
                rr.LineStrips3D.from_fields(colors=[150, 150, 150], radii=rr.Radius.ui_points(1)),
                static=True,
            )
            recording.send_columns(
                "/encounter/port_gap_line",
                indexes=indexes,
                columns=rr.LineStrips3D.columns(strips=port_positions),
            )
        recording.log(
            "/encounter/relative_path",
            rr.LineStrips3D(local_position, colors=CHASER_COLOR, radii=0.05),
            static=True,
        )
        recording.log(
            "/encounter/axes",
            rr.Arrows3D(
                vectors=np.eye(3) * 8,
                colors=AXIS_COLORS,
            ),
            static=True,
        )
        recording.log(
            "/encounter/axes/labels",
            rr.Points3D(
                np.eye(3) * 8,
                radii=0,
                colors=AXIS_COLORS,
                labels=["+X", "+Y", "+Z"],
                show_labels=True,
            ),
            static=True,
        )
        recording.log(
            "/encounter/chaser/axes",
            rr.Arrows3D(vectors=np.eye(3) * 2, colors=AXIS_COLORS),
            static=True,
        )

        series = [
            ("range_m", "range" if control is None else "COM range", np.linalg.norm(local_position, axis=-1), TARGET_COLOR),
            ("speed_mps", "speed", np.linalg.norm(local_velocity, axis=-1), CHASER_COLOR),
        ]
        series.extend(
            (f"position_m/{axis}", axis, local_position[:, index], AXIS_COLORS[index])
            for index, axis in enumerate(("x", "y", "z"))
        )
        for entity, label, values, color in series:
            _log_series(recording, entity, label, values, color, indexes)
        if control is not None:
            _log_control(recording, control, times, manifest["body_order"])
        if manifest.get("schema_version") == 3:
            if observations is None or estimates is None or image_root is None:
                raise ValueError("visual replay requires saved observations and navigation estimates")
            _log_visual(
                recording, trajectory, basis_nl, observations, estimates, Path(image_root)
            )
            if lidar:
                _log_lidar(recording, manifest, trajectory, basis_nl, lidar, Path(image_root))
        for event in events or ():
            recording.set_time("sim_time", duration=np.timedelta64(event["time_ns"], "ns"))
            recording.log(
                "/events",
                rr.TextLog(
                    f"{event['kind']}: {event['reason']} (phase={PHASE_NAMES[event['phase']]})",
                    level="INFO",
                ),
            )
        recording.flush()
    finally:
        recording.disconnect()
    return output


def _log_visual(recording, trajectory, basis_nl, observations, estimates, image_root):
    """Display already delivered observations and saved estimates on physical time."""
    from .frames import quaternion_matrices
    from .vision_records import read_rgb

    estimated_color = [220, 110, 240]
    for name, label in (
        ("age_s", "last accepted capture age"),
        ("health", "navigation health"),
        ("position_error_m", "estimated target position error"),
        ("attitude_error_deg", "estimated target attitude error"),
    ):
        recording.log(f"/vision/{name}", rr.SeriesLines(names=label), static=True)
    for index, row in enumerate(estimates):
        recording.set_time("sim_time", duration=np.timedelta64(row["time_ns"], "ns"))
        target = row["target"]
        if target is None:
            recording.log("/encounter/estimated_target", rr.Clear(recursive=True))
            recording.log("/vision/health", rr.Scalars(-1))
            for entity in ("age_s", "position_error_m", "attitude_error_deg"):
                recording.log(f"/vision/{entity}", rr.Clear(recursive=False))
            continue
        estimate_position = np.asarray(target["position_m"])
        estimate_rotation = quaternion_matrices(target["quaternion_xyzw"])
        truth_rotation = quaternion_matrices(trajectory["quaternions_xyzw"][index, 0])
        position_error = estimate_position - trajectory["positions_m"][index, 0]
        cosine = (np.trace(truth_rotation.T @ estimate_rotation) - 1) / 2
        recording.log(
            "/encounter/estimated_target",
            rr.Boxes3D(
                sizes=[2, 2, 2], colors=estimated_color,
                fill_mode=rr.components.FillMode.MajorWireframe,
                labels=["estimated target"], show_labels=True,
            ),
            rr.Transform3D(
                translation=basis_nl[index].T @ position_error,
                mat3x3=basis_nl[index].T @ estimate_rotation,
            ),
        )
        recording.log("/vision/age_s", rr.Scalars(
            (row["time_ns"] - target["capture_time_ns"]) / 1e9
        ))
        recording.log("/vision/health", rr.Scalars(
            {"fresh": 0, "predicted": 1, "lost": 2}[target["health"]]
        ))
        recording.log("/vision/position_error_m", rr.Scalars(np.linalg.norm(position_error)))
        recording.log("/vision/attitude_error_deg", rr.Scalars(
            np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        ))
    for row in observations:
        delivery = row["actual_delivery_time_ns"]
        if delivery is None:
            continue
        recording.set_time("sim_time", duration=np.timedelta64(delivery, "ns"))
        pose = row["pose_estimate"]
        if row["image_path"] is None:
            recording.log("/camera/rgb", rr.Clear(recursive=True))
        else:
            recording.log("/camera/rgb", rr.Image(read_rgb(image_root / row["image_path"])))
        if pose.get("image_corners") is None:
            recording.log("/camera/corners", rr.Clear(recursive=True))
        else:
            recording.log("/camera/corners", rr.Points2D(
                pose["image_corners"], radii=3, colors=estimated_color,
            ))
        recording.log("/vision/status", rr.TextLog(
            f"capture={row['capture_id']} captured={row['capture_time_ns']} ns "
            f"delivered={delivery} ns pose_valid={pose['valid']} "
            f"navigation_accepted={row['navigation_accepted']} reason={pose.get('reason', '')}"
        ))


def _log_lidar(recording, manifest, trajectory, basis_nl, rows, root):
    """Show each delivered scan in target LVLH at its capture pose; every fourth return
    plus all retroreflector returns."""
    from .frames import quaternion_matrices
    from .lidar import LidarModel

    model = LidarModel(**manifest["lidar"]["model"])
    directions = model.directions()
    sampled = np.zeros(directions.shape[:2], dtype=bool)
    sampled[::2, ::2] = True
    for row in rows:
        if row["actual_delivery_time_ns"] is None:
            continue
        recording.set_time("sim_time", duration=np.timedelta64(row["actual_delivery_time_ns"], "ns"))
        with np.load(root / row["scan_path"], allow_pickle=False) as scan:
            ranges, retro = scan["ranges_m"], scan["retroreflective"]
        shown = np.isfinite(ranges) & (sampled | retro)
        tick = row["capture_time_ns"] // manifest["dt_ns"]
        points_body = (np.asarray(model.mount_position_body_m)
                       + (directions[shown] * ranges[shown, None]) @ np.asarray(model.rotation_bc).T)
        points_n = (trajectory["own_position_m"][tick]
                    + points_body @ quaternion_matrices(trajectory["own_quaternion_xyzw"][tick]).T)
        recording.log("/encounter/lidar", rr.Points3D(
            (points_n - trajectory["positions_m"][tick, 0]) @ basis_nl[tick],
            colors=np.where(retro[shown, None], [255, 210, 40], [70, 200, 220]),
            radii=0.015,
        ))
        pose = row["pose_estimate"]
        recording.log("/vision/status", rr.TextLog(
            f"lidar scan={row['capture_id']} captured={row['capture_time_ns']} ns "
            f"reflectors={pose.get('reflector_count', 0)} pose_valid={pose['valid']} "
            f"navigation_accepted={row['navigation_accepted']} reason={pose.get('reason', '')}"
        ))


def _log_series(recording, entity, label, values, color, indexes, *, step=False) -> None:
    path = f"/telemetry/{entity}"
    recording.log(
        path,
        rr.SeriesLines(
            colors=color,
            names=label,
            interpolation_mode=rr.components.InterpolationMode.StepAfter if step else None,
        ),
        static=True,
    )
    recording.send_columns(path, indexes=indexes, columns=rr.Scalars.columns(scalars=values))


def _log_control(recording, control: dict, times: np.ndarray, body_order: list) -> None:
    indexes = [rr.TimeColumn("sim_time", duration=times)]
    for entity, label, values, color in (
        ("port_gap_m", "port axial gap", control["port_gap_m"], CHASER_COLOR),
        ("desired_gap_m", "desired gap", control["desired_gap_m"], [160, 160, 160]),
        ("alignment_deg", "alignment", control["alignment_deg"], CHASER_COLOR),
        ("phase", "phase", control["phase"], TARGET_COLOR),
        ("port_velocity_mps/speed", "speed", np.linalg.norm(control["port_velocity_mps"], axis=-1), [150, 150, 150]),
    ):
        _log_series(recording, entity, label, values, color, indexes, step=entity == "phase")
    for axis, name in enumerate(("x", "y", "z")):
        _log_series(
            recording, f"port_velocity_mps/{name}", name,
            control["port_velocity_mps"][:, axis], AXIS_COLORS[axis], indexes,
        )
    # A command belongs to [tick, tick+1), with no invented command at the final state.
    if len(times) > 1:
        command_indexes = [rr.TimeColumn("sim_time", duration=times[:-1])]
        for quantity, unit in (("forces", "n"), ("torques", "nm")):
            entity = "force_n" if quantity == "forces" else "torque_nm"
            for body, name in enumerate(body_order):
                for kind in ("requested", "applied"):
                    color = (TARGET_COLOR, CHASER_COLOR)[body]
                    if kind == "requested":
                        color = [int((value + 255) / 2) for value in color]
                    _log_series(
                        recording, f"{entity}/{name}/{kind}", f"{name} {kind}",
                        control[f"{kind}_{quantity}_{unit}"][:, body], color, command_indexes,
                        step=True,
                    )


def open_replay(path: Path) -> None:
    """Launch the SDK's bundled native viewer through its supported CLI entrypoint."""
    path = Path(path).resolve(strict=True)
    subprocess.Popen(
        [sys.executable, "-m", "rerun_cli", "--detach-process", str(path)],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

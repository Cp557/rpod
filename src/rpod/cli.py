"""Command-line entry points for orbital and controlled approach experiments."""

import argparse
import json
from pathlib import Path
import shutil
import sys

from .scenarios import SCENARIOS

# The qualified realistic-scene camera calibration, which every suite from M8 on uses.
REALISTIC_CALIBRATION = Path("datasets/m8-vbar/calibration.json")
# The navigation surrogate every published Monte Carlo campaign used.
PUBLISHED_SURROGATE = Path("models/m9-surrogate.json")


def _suite_parser(commands, name, description, label):
    """A suite subcommand with the shared --output, --calibration and --label options."""
    parser = commands.add_parser(name, help=description)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, default=REALISTIC_CALIBRATION)
    parser.add_argument("--label", default=label, help="evidence name in benchmarks/")
    return parser


def _publish(report, name):
    """Copy a run's report into benchmarks/ as `name`, the evidence the docs cite."""
    evidence = Path("benchmarks") / name
    evidence.parent.mkdir(exist_ok=True)
    shutil.copyfile(report, evidence)
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rpod", description="Orbital and controlled spacecraft experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run and record a headless spacecraft experiment")
    run.add_argument("--scenario", choices=SCENARIOS, default="circular")
    run.add_argument("--output", required=True, type=Path)
    replay = commands.add_parser(
        "replay", help="rebuild a replay from saved numerical records"
    )
    replay.add_argument("directory", type=Path)
    replay.add_argument(
        "--save-only", action="store_true", help="export without opening the viewer"
    )
    benchmark = commands.add_parser(
        "benchmark", help="measure release-build stepping and recording"
    )
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--batch", action="store_true",
                           help="measure batched stepping, scaling and the closed-loop fleet")
    camera = commands.add_parser(
        "camera-check", help="verify calibrated Blender projection and raster alignment"
    )
    camera.add_argument("--output", required=True, type=Path)
    camera.add_argument("--blender", type=Path)
    perception = commands.add_parser(
        "perception-eval", help="render and evaluate cooperative-target pose estimates"
    )
    perception_source = perception.add_mutually_exclusive_group(required=True)
    perception_source.add_argument("--suite", choices=("m3",))
    perception_source.add_argument("--source", type=Path)
    perception.add_argument("--output", required=True, type=Path)
    perception.add_argument("--blender", type=Path)
    perception.add_argument("--scene", choices=("simple", "realistic"), default="simple")
    perception.add_argument(
        "--earth", choices=("vbar", "rbar"), default="vbar",
        help="realistic fixtures: Earth below the image (vbar) or behind the target (rbar)",
    )
    visual = commands.add_parser("visual-run", help="run a camera-controlled terminal approach")
    visual.add_argument(
        "--case",
        choices=("nominal", "disturbed", "hidden", "abort", "short-loss", "long-loss"),
        default="nominal",
    )
    visual.add_argument("--seed", type=int, default=0)
    visual.add_argument("--calibration", type=Path, default=Path("datasets/m3/calibration.json"))
    visual.add_argument("--output", required=True, type=Path)
    suite = commands.add_parser(
        "m5-suite", help="run or resume the frozen 40-episode M5 visual evaluation"
    )
    suite.add_argument("--calibration", type=Path, default=Path("datasets/m3/calibration.json"))
    suite.add_argument("--output", required=True, type=Path)
    suite.add_argument("--lidar", action="store_true", help="add the flash lidar and fuse it")
    suite.add_argument("--scene", choices=("simple", "realistic"), default="simple")
    suite.add_argument("--controller", choices=("reference", "learned"), default="reference")
    suite.add_argument("--policy", type=Path, default=Path("models/m10-policy.npy"))
    suite.add_argument("--label", help="evidence name in benchmarks/ (default from the scene)")
    showcase = commands.add_parser(
        "showcase", help="render a realistic outside-view movie of a saved visual run"
    )
    showcase.add_argument("directory", type=Path)
    showcase.add_argument("--output", required=True, type=Path)
    fit = commands.add_parser(
        "fit-surrogate", help="fit and validate the navigation surrogate on a recorded suite"
    )
    fit.add_argument("--suite", type=Path, default=Path("runs/m8-realistic"))
    fit.add_argument("--output", type=Path, default=Path("models/m9-surrogate.json"))
    fit.add_argument("--draws", type=int, default=200)
    fit.add_argument("--scenario", choices=("operational", "tumbling"), default="operational")
    campaign = commands.add_parser(
        "monte-carlo", help="run a batched Monte Carlo campaign with the navigation surrogate"
    )
    campaign.add_argument("--episodes", type=int, default=100_000)
    campaign.add_argument("--seed", type=int, default=9)
    campaign.add_argument("--model", type=Path, default=PUBLISHED_SURROGATE)
    campaign.add_argument("--controller", choices=("reference", "learned"), default="reference")
    campaign.add_argument("--scenario", choices=("operational", "tumbling"),
                          default="operational")
    campaign.add_argument("--policy", type=Path, default=Path("models/m10-policy.npy"))
    campaign.add_argument("--output", required=True, type=Path)
    replay_episode = commands.add_parser(
        "monte-carlo-replay", help="rerun one campaign episode with full rendering"
    )
    replay_episode.add_argument("campaign", type=Path)
    replay_episode.add_argument("--index", type=int, required=True)
    replay_episode.add_argument("--calibration", type=Path, default=REALISTIC_CALIBRATION)
    replay_episode.add_argument("--output", required=True, type=Path)
    training = commands.add_parser(
        "train-policy", help="train the learned approach policy with evolution strategies"
    )
    training.add_argument("--output", type=Path, default=Path("models/m10-policy.npy"))
    training.add_argument("--model", type=Path, default=Path("models/m9-surrogate.json"))
    training.add_argument("--generations", type=int, default=300)
    training.add_argument("--pairs", type=int, default=64)
    training.add_argument("--episodes", type=int, default=16)
    training.add_argument("--validation-episodes", type=int, default=1024)
    commands.add_parser("m11-assets", help="export the unmarked mesh, choose keypoints and "
                                          "calibrate lidar ICP")
    dataset = commands.add_parser("keypoint-dataset", help="render labelled keypoint images")
    dataset.add_argument("--output", type=Path, default=Path("datasets/m11/keypoints"))
    dataset.add_argument("--train", type=int, default=9000)
    dataset.add_argument("--val", type=int, default=1000)
    dataset.add_argument("--test", type=int, default=1500)
    dataset.add_argument("--calibration", type=Path, default=REALISTIC_CALIBRATION)
    keypoint_calibration = commands.add_parser(
        "keypoint-calibrate", help="evaluate the keypoint network and write its calibration")
    keypoint_calibration.add_argument("--dataset", type=Path,
                                      default=Path("datasets/m11/keypoints"))
    keypoint_calibration.add_argument("--model", type=Path, required=True)
    keypoint_calibration.add_argument("--calibration", type=Path, default=REALISTIC_CALIBRATION)
    keypoint_calibration.add_argument("--output", type=Path,
                                      default=Path("models/m11-camera-calibration.json"))
    m11 = _suite_parser(commands, "m11-suite", "run or resume the M11 tumbling-target suite",
                        "m11")
    m11.add_argument("--keypoint-model", type=Path, help="fuse camera keypoints from this model")
    m12 = _suite_parser(commands, "m12-suite", "run or resume the M12 lidar-fault suite", "m12")
    m12.add_argument("--keypoint-model", type=Path, help="fuse camera keypoints from this model")
    m12.add_argument("--configuration", choices=("m11", "robust"), default="m11")
    commands.add_parser("m13-calibrate", help="calibrate the M13 docking lidar against truth")
    _suite_parser(commands, "m13-suite", "run or resume the M13 docking suite", "m13")
    m14 = _suite_parser(commands, "m14-suite", "run or resume the M14 thruster suite", "m14")
    m14.add_argument("--seed", type=int, default=11,
                     help="11 is the suite; any other seed gives development episodes")
    m14.add_argument("--episodes", nargs="+", help="run only these episode ids")
    stage0 = commands.add_parser(
        "m15-stage0", help="measure the M15 far-range unknowns: physics cost and accuracy, "
                           "pointing, eclipses and detectability renders"
    )
    stage0.add_argument("--output", required=True, type=Path)
    stage0.add_argument("--no-renders", action="store_true", help="skip the GPU renders")
    passive = commands.add_parser(
        "passive-safety", help="coast from every 10 s of a tumbling suite's operational episodes"
    )
    passive.add_argument("run", type=Path)
    passive.add_argument("--output", required=True, type=Path)
    cases = commands.add_parser(
        "basilisk-cases", help="write dynamics cross-check cases for tools/basilisk"
    )
    cases.add_argument("--output", required=True, type=Path)
    crosscheck = commands.add_parser(
        "basilisk-compare", help="compare rpod dynamics with Basilisk trajectories"
    )
    crosscheck.add_argument("--cases", required=True, type=Path)
    crosscheck.add_argument("--basilisk", required=True, type=Path)
    crosscheck.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .recording import run_scenario

            result = run_scenario(args.scenario, args.output)
            print(f"Saved {result['samples']} states and replay to {args.output.resolve()}")
            if "outcome" in result:
                print(
                    f"Outcome: {result['outcome']}; final port gap {result['final_gap_m']:.3f} m"
                )
        elif args.command == "replay":
            from .recording import replay_run

            print(f"Replay: {replay_run(args.directory, args.save_only).resolve()}")
        elif args.command == "benchmark":
            from .benchmark import run_batch_benchmark, run_benchmark

            (run_batch_benchmark if args.batch else run_benchmark)(args.output)
            print(f"Benchmark: {args.output.resolve()}")
        elif args.command == "camera-check":
            from .vision_eval import camera_check

            report = camera_check(args.output, args.blender)
            print(
                f"Camera check passed: {report['qualification_passed']}; "
                f"projection {report['projection_max_error_px']:.6f} px; "
                f"raster {report['raster_max_error_px']:.6f} px"
            )
            return 0 if report["qualification_passed"] else 1
        elif args.command == "perception-eval":
            from .vision_eval import perception_eval

            report = perception_eval(args.output, args.source, args.blender, args.scene, args.earth)
            print(f"Perception report: {args.output.resolve() / 'report.json'}")
            if "qualification_passed" in report:
                print(f"M3-fixture qualification passed ({args.scene} scene): "
                      f"{report['qualification_passed']}")
                return 0 if report["qualification_passed"] else 1
        elif args.command == "m5-suite":
            from .m5_suite import run_suite

            report = run_suite(args.output, args.calibration, args.lidar, args.scene,
                               args.controller, args.policy)
            if args.scene == "realistic":
                milestone, name = "M8", "m8-realistic" + ("-lidar" if args.lidar else "")
            else:
                milestone, name = ("M6", "m6-lidar") if args.lidar else ("M5", "m5")
            if args.label:
                milestone, name = args.label, args.label
            evidence = _publish(args.output / "report.json", f"{name}-evaluation.json")
            print(f"{milestone} passed: {report['passed']}; report: {evidence.resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "fit-surrogate":
            from .monte_carlo import OPERATIONAL, fit_and_validate

            if args.scenario == "tumbling":
                from .tumble_fleet import tumbling_scenario

                model = fit_and_validate(args.suite, args.output, args.draws,
                                         tumbling_scenario(), failures="updates")
            else:
                model = fit_and_validate(args.suite, args.output, args.draws, OPERATIONAL)
            print(f"Surrogate: {args.output.resolve()}; validation passed: "
                  f"{model['validation']['passed']}")
            return 0 if model["validation"]["passed"] else 1
        elif args.command == "monte-carlo":
            from .monte_carlo import OPERATIONAL, run_campaign
            from .vision_records import sha256_file

            # The campaign hashes the fleet, surrogate and campaign code; record the scenario's
            # and controller's own code too.
            source, provenance = Path(__file__).parent, {}
            scenario = OPERATIONAL
            if args.scenario == "tumbling":
                from .tumble_fleet import tumbling_scenario

                scenario = tumbling_scenario()
                provenance["scenario_implementation_sha256"] = {
                    name: sha256_file(source / name) for name in ("tumble_fleet.py", "tumble.py")}

            factory = None
            if args.controller == "learned":
                from .learned_control import IMPLEMENTATION_FILES, LearnedFleetController
                from .policy import load_policy

                parameters = load_policy(args.policy)
                provenance.update(policy=args.policy.as_posix(),
                                  policy_sha256=sha256_file(args.policy),
                                  controller_implementation_sha256={
                                      name: sha256_file(source / name)
                                      for name in IMPLEMENTATION_FILES})

                def factory(mission, count):
                    return LearnedFleetController(mission, count, parameters)
            report = run_campaign(
                args.output, args.episodes, args.seed, args.model, controller_factory=factory,
                label=args.controller, scenario=scenario, provenance=provenance)
            name = ("m11-monte-carlo" if args.scenario == "tumbling" else
                    "m9-monte-carlo" if args.controller == "reference" else "m10-monte-carlo")
            evidence = args.output / "report.json"
            # Only the published settings replace the evidence in benchmarks/.
            if (args.episodes, args.seed, args.model) == (100_000, 9, PUBLISHED_SURROGATE):
                evidence = _publish(evidence, f"{name}.json")
            low, high = report["success_fraction_95_interval"]
            print(f"Success {report['success_fraction']:.5f} (95% interval {low:.5f}-{high:.5f}) "
                  f"over {report['episodes']} episodes; report: {evidence.resolve()}")
        elif args.command == "monte-carlo-replay":
            from .monte_carlo import replay_episode

            summary = replay_episode(args.campaign, args.index, args.calibration, args.output)
            print(f"Rendered outcome: {summary['outcome']}; "
                  f"report: {args.output.resolve() / 'summary.json'}")
        elif args.command == "train-policy":
            from .train_policy import train

            selected = train(args.output, args.model, args.generations, args.pairs,
                             args.episodes, validation_episodes=args.validation_episodes)
            print(f"Policy: {args.output.resolve()}; selected generation "
                  f"{selected['generation']}, validation success "
                  f"{selected['success_fraction']:.4f}")
        elif args.command == "m11-assets":
            from .m11_suite import prepare_assets

            print(f"M11 assets ready; removed cooperative aids: {', '.join(prepare_assets())}")
        elif args.command == "keypoint-dataset":
            from .keypoints import load_keypoints, render_dataset
            from .m11_suite import KEYPOINTS
            from .target_mesh import MESH_PATH, UnmarkedRenderer, load_target_mesh
            from .visual_run import camera_from_settings, load_qualified_calibration

            camera = camera_from_settings(load_qualified_calibration(args.calibration)["camera"])
            render_dataset(args.output, {"train": args.train, "val": args.val, "test": args.test},
                           11, camera, load_target_mesh(MESH_PATH), load_keypoints(KEYPOINTS),
                           lambda folder: UnmarkedRenderer(folder, camera=camera))
            print(f"Keypoint dataset: {args.output.resolve()}")
        elif args.command == "keypoint-calibrate":
            from .keypoints import (KeypointEstimator, calibrate_keypoints, error_summary,
                                    evaluate, load_keypoints, model_provenance)
            from .m11_suite import KEYPOINTS
            from .recording import write_json
            from .visual_run import camera_from_settings, load_qualified_calibration

            settings = load_qualified_calibration(args.calibration)["camera"]
            camera = camera_from_settings(settings)
            estimator = KeypointEstimator(args.model, load_keypoints(KEYPOINTS), camera)
            bins = [5.7, 8.45, 15.95, 25.95, 31.95, 1e9]
            provenance = model_provenance(args.model, KEYPOINTS)
            validation = evaluate(args.dataset, "val", estimator)
            calibration = calibrate_keypoints(validation, bins, settings, provenance)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8",
                                   newline="\n")
            test = evaluate(args.dataset, "test", estimator, seed=1)
            evidence = Path("benchmarks") / "m11-keypoint-evaluation.json"
            evidence.parent.mkdir(exist_ok=True)
            write_json(evidence, {
                "artifact_type": "m11_keypoint_evaluation", **provenance,
                "calibration_qualified": calibration["qualification_passed"],
                "gross_errors": calibration["gross_errors"],
                "validation_by_range": error_summary(validation, bins),
                "held_out_test_by_range": error_summary(test, bins),
                "held_out_conditions": "Sun phase 70-100 degrees, 3 DN / 0.75 px degradation",
                "held_out_rows": test})
            valid = [row for row in test if row["valid"]]
            print(f"Keypoint calibration: {args.output.resolve()}; qualified "
                  f"{calibration['qualification_passed']}; held-out valid "
                  f"{len(valid)}/{len(test)}; report: {evidence.resolve()}")
            return 0 if calibration["qualification_passed"] else 1
        elif args.command == "m11-suite":
            from .m11_suite import run_m11_suite

            report = run_m11_suite(args.output, args.calibration, args.keypoint_model)
            evidence = _publish(args.output / "report.json", f"{args.label}-evaluation.json")
            print(f"M11 passed: {report['passed']}; report: {evidence.resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "m12-suite":
            from .m12_suite import run_m12_suite

            report = run_m12_suite(args.output, args.calibration, args.keypoint_model,
                                   args.configuration)
            evidence = _publish(args.output / "report.json", f"{args.label}-evaluation.json")
            print(f"M12 passed: {report['passed']}; recovered {report['recovered']} of "
                  f"{len(report['episodes'])}; report: {evidence.resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "m13-calibrate":
            from .m13_suite import LIDAR_CALIBRATION, prepare_docking_calibration

            calibration = prepare_docking_calibration()
            print(f"Docking lidar calibration: {LIDAR_CALIBRATION.resolve()}; samples per bin "
                  f"{calibration['samples']}")
        elif args.command == "m13-suite":
            from .m13_suite import run_m13_suite

            report = run_m13_suite(args.output, args.calibration)
            evidence = _publish(args.output / "report.json", f"{args.label}-evaluation.json")
            print(f"M13 passed: {report['passed']}; docked {report['docked']} of "
                  f"{len(report['episodes'])}; report: {evidence.resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "m14-suite":
            from .m14_suite import run_m14_suite

            report = run_m14_suite(args.output, args.calibration, args.seed, args.episodes)
            if args.seed == 11 and args.episodes is None:
                _publish(args.output / "report.json", f"{args.label}-evaluation.json")
            print(f"M14 passed: {report['passed']}; docked {report['docked']} of "
                  f"{len(report['episodes'])}; report: {(args.output / 'report.json').resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "m15-stage0":
            from .m15_stage0 import run_stage0

            run_stage0(args.output, renders=not args.no_renders)
            report = args.output / "report.json"
            if not args.no_renders:
                report = _publish(report, "m15-stage0.json")
            print(f"M15 stage 0 report: {report.resolve()}")
        elif args.command == "passive-safety":
            from .lidar_faults import passive_safety_map
            from .recording import write_json
            from .target_mesh import MESH_PATH, load_target_mesh

            result = passive_safety_map(args.run, load_target_mesh(MESH_PATH))
            write_json(args.output, result)
            print(f"Passively safe from {result['passively_safe']} of {result['samples']} "
                  f"states; report: {args.output.resolve()}")
        elif args.command == "basilisk-cases":
            from .crosscheck import basilisk_cases

            basilisk_cases(args.output)
            print(f"Cases: {args.output.resolve()}")
        elif args.command == "basilisk-compare":
            from .crosscheck import compare

            report = compare(args.cases, args.basilisk, args.output)
            for name, row in report["cases"].items():
                print(f"{name}: position {row['max_position_m']:.2e} m, attitude "
                      f"{row['max_attitude_rad']:.2e} rad, passed {row['passed']}")
            return 0 if report["passed"] else 1
        elif args.command == "showcase":
            from .showcase import render_showcase

            print(f"Movie: {render_showcase(args.directory, args.output).resolve()}")
        else:
            from .visual_run import run_visual

            report = run_visual(args.case, args.seed, args.output, args.calibration)
            print(
                f"Visual outcome: {report['outcome']}; "
                f"report: {args.output.resolve() / 'summary.json'}"
            )
            return 0 if report.get("qualification_passed", False) else 1
    except (OSError, ValueError, RuntimeError, ImportError, OverflowError) as error:
        print(f"rpod: {error}", file=sys.stderr)
        return 1
    return 0

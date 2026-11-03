"""Command-line entry points for orbital and controlled approach experiments."""

import argparse
from pathlib import Path
import sys

from .scenarios import SCENARIOS


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
    campaign = commands.add_parser(
        "monte-carlo", help="run a batched Monte Carlo campaign with the navigation surrogate"
    )
    campaign.add_argument("--episodes", type=int, default=100_000)
    campaign.add_argument("--seed", type=int, default=9)
    campaign.add_argument("--model", type=Path, default=Path("models/m9-surrogate.json"))
    campaign.add_argument("--output", required=True, type=Path)
    replay_episode = commands.add_parser(
        "monte-carlo-replay", help="rerun one campaign episode with full rendering"
    )
    replay_episode.add_argument("campaign", type=Path)
    replay_episode.add_argument("--index", type=int, required=True)
    replay_episode.add_argument("--calibration", type=Path,
                                default=Path("datasets/m8-vbar/calibration.json"))
    replay_episode.add_argument("--output", required=True, type=Path)
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
            import shutil

            from .m5_suite import run_suite

            report = run_suite(args.output, args.calibration, args.lidar, args.scene)
            if args.scene == "realistic":
                milestone, name = "M8", "m8-realistic" + ("-lidar" if args.lidar else "")
            else:
                milestone, name = ("M6", "m6-lidar") if args.lidar else ("M5", "m5")
            evidence = Path("benchmarks") / f"{name}-evaluation.json"
            evidence.parent.mkdir(exist_ok=True)
            shutil.copyfile(args.output / "report.json", evidence)
            print(f"{milestone} passed: {report['passed']}; report: {evidence.resolve()}")
            return 0 if report["passed"] else 1
        elif args.command == "fit-surrogate":
            from .monte_carlo import fit_and_validate

            model = fit_and_validate(args.suite, args.output, args.draws)
            print(f"Surrogate: {args.output.resolve()}; validation passed: "
                  f"{model['validation']['passed']}")
            return 0 if model["validation"]["passed"] else 1
        elif args.command == "monte-carlo":
            import shutil

            from .monte_carlo import run_campaign

            report = run_campaign(args.output, args.episodes, args.seed, args.model)
            evidence = Path("benchmarks") / "m9-monte-carlo.json"
            evidence.parent.mkdir(exist_ok=True)
            shutil.copyfile(args.output / "report.json", evidence)
            low, high = report["success_fraction_95_interval"]
            print(f"Success {report['success_fraction']:.5f} (95% interval {low:.5f}-{high:.5f}) "
                  f"over {report['episodes']} episodes; report: {evidence.resolve()}")
        elif args.command == "monte-carlo-replay":
            from .monte_carlo import replay_episode

            summary = replay_episode(args.campaign, args.index, args.calibration, args.output)
            print(f"Rendered outcome: {summary['outcome']}; "
                  f"report: {args.output.resolve() / 'summary.json'}")
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

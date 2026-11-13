# M9: batched engine, navigation surrogate and Monte Carlo

The 40-episode suites show the autonomy succeeding on chosen conditions. They cannot say how often it fails. Bounding a failure rate near 1 in 1,000 takes about 100,000 episodes, and a rendered episode runs in about real time (210 s), so the same campaign rendered would take 240 days of GPU time. M9 makes the question answerable in minutes:

1. step many episodes at once in Rust;
2. run the reference autonomy on all of them at once in NumPy;
3. stand in for rendering, perception and the filter with a statistical surrogate, fitted to rendered episodes and checked against episodes it did not see.

[REFERENCE.md](../REFERENCE.md#batched-engine-and-monte-carlo-m9) is the full specification; this note covers the method, the evidence and the limits.

## Method

**Batched engine.** `RigidBodyBatch` holds one `RigidBodyEngine` per environment with its own rayon thread pool. Environments share nothing, so a batch is bitwise equal to independent engines for any thread count (Rust tests), and a failing environment stops alone. Pair geometry for every environment (`pair_metrics`) is also computed in Rust, in one pass into one array.

**Vectorized autonomy.** `fleet.py` copies the reference behavior across a batch: the force and torque law, the whole supervisor (acquisition, braking, reacquisition, loss, aborts, completion) and the environment's timing, guards and scoring. Tests hold it equal to the scalar code: the law within 1e-11, every tick's phase and command over scripted sequences, and the paired oracle's outcome and end tick.

**Navigation surrogate.** The fleet has no renderer, so `surrogate.py` replays what the realistic M8 suite's filter produced instead:

- estimate errors at each camera delivery, by range for position and velocity and by time since the first estimate for attitude and rate, which converge;
- correlated (AR(1)) residuals between deliveries, with position drifting by the velocity error as the filter's prediction does;
- rejected deliveries drawn by time since the first estimate, which hold the estimate and make the controller brake exactly as the reference client does.

**Validation before use.** The surrogate is fitted on half of M8's 28 operational episodes and must reproduce the other half: for each of completion time, minimum port gap, final port speed and braking count, at least 25 of the 28 rendered values must fall inside the central 95% band of 200 surrogate runs of the same episode, and estimate-error medians and P95 must agree within 25%. Only then is the model fitted on all 28 and used.

## Results

All M9 checks pass.

| Surrogate check (28 rendered episodes) | Inside the surrogate's 95% band |
| --- | --- |
| Completion time | 27 |
| Minimum port gap | 27 |
| Final port speed | 26 |
| Braking count | 28 |

Held-out estimate errors agree within 4–5% (position) and 12–15% (attitude).

**100,000-episode campaign** (reference controller, seed 9, 556 s on a 6-core desktop):

| Result | Value |
| --- | --- |
| Success | 99.932% (95% Clopper–Pearson interval 99.914–99.947%) |
| Failures | 68, all `navigation_lost`; no false completion, boundary violation or abort |
| Completion time | median 210.2 s, P99 228.6 s |
| Closest approach | 4.920 m port to port |
| Final port speed | at most 3.4 mm/s |
| Δv | median 0.365 m/s, P99 0.464 m/s |

Every failure is the supervisor's designed safe response to two consecutive rejected deliveries in the first 30 s. Rerunning the seed reproduces every episode exactly, and `rpod monte-carlo-replay` reruns any one of them as a full rendered episode.

**Throughput.** The batch steps 15.7 million environment ticks per second on 12 threads (2.2 million on one; 7.2× over 6 physical cores) at 632 bytes of engine state per environment. With the autonomy in the loop, the fleet runs about 670,000 environment ticks per second, about 900 times the scalar runner.

**Independent check.** Basilisk 2.12.2, integrating the same equations with the same RK4 step, agrees with the engine to round-off: 1.5e-8 m after one orbit, 9.4e-13 rad after 1,000 s of tumbling.

## Limitations

- **The surrogate is only as good as its 28 episodes.** It reproduces the conditions M8 rendered; it says nothing about lighting, ranges or faults outside them, and the rendered suite remains the qualification.
- **The failure rate is the least constrained number.** It comes from rejection probabilities estimated from 4 rejections in about 5,900 deliveries and treated as independent. The interval covers sampling noise in the campaign, not uncertainty in the fitted model.
- **The fleet does not render, so it cannot find perception failures**; it finds how the supervisor responds to the perception errors the surrogate draws.
- Timings come from a desktop that was also rendering, not an isolated machine.

## Reproduce

```bash
uv run --no-sync rpod fit-surrogate --suite runs/m8-realistic --output models/m9-surrogate.json
uv run --no-sync rpod monte-carlo --episodes 100000 --seed 9 --output runs/m9-monte-carlo
uv run --no-sync rpod monte-carlo-replay runs/m9-monte-carlo --index 6 --output runs/m9-replay-6
uv run --no-sync rpod benchmark --batch --output runs/m9-batch.json
uv run --no-sync rpod basilisk-cases --output runs/m9-crosscheck/cases.json
uv run --project tools/basilisk python tools/basilisk/propagate.py runs/m9-crosscheck/cases.json runs/m9-crosscheck/basilisk.json
uv run --no-sync rpod basilisk-compare --cases runs/m9-crosscheck/cases.json --basilisk runs/m9-crosscheck/basilisk.json --output runs/m9-crosscheck/report.json
```

`fit-surrogate` needs the rendered M8 suite (`runs/m8-realistic`); the campaign needs only the fitted model, which is tracked in `models/`.

# M14: thrusters and thruster faults

Through M13 the chaser pushed itself with an ideal force and torque: any direction, any size up to 2 N and 0.5 N·m, delivered exactly. Real spacecraft fire fixed thrusters in short pulses, their thrust is never quite what the data sheet says, and a valve can stick. M14 flies the M13 docking on 16 thrusters and fails one of them at the worst moments: mid-approach, at the hold, and in the last meter before contact. The thruster that fails is whichever in its group (closing, or sideways) had been working hardest in the previous 10 seconds.

[REFERENCE.md](../REFERENCE.md#thrusters-and-thruster-faults-m14) is the full specification.

## The thruster set

Sixteen 1 N thrusters sit in four corner pods, the layout the showcase movie already drew. Each pod fires outward sideways, outward up or down, and both ways along the docking axis.
- **Specific impulse** is 220 s, and the **minimum on-time** is 10 ms.
- **The hardware is imperfect:** each thruster's thrust is off by up to 3% and its direction by up to 0.5°, drawn per episode. The flight software knows only the nominal design.
- **Authority:** with everything healthy, the set delivers ±2 N sideways, ±4 N along the docking axis, and ±1.44 N·m about every axis.

## The thruster driver

The controller still asks for a force and a torque. An onboard driver turns that into valve commands every 0.1 s tick, and watches for failures.

- **Allocation is a linear program.** Find duty cycles in [0, 1] that produce the requested force and torque exactly, using the least total firing time, which is the least propellant.
  - If the request is beyond what the healthy thrusters can do, the driver delivers the largest fraction of it in the same direction instead.
  - With all 16 healthy, every request M13's controller could make is met exactly.
  - After losing any one thruster, every axis keeps at least 0.72 N or N·m. Six-axis control survives any single failure, so the response to a failure is to isolate it and carry on.
- **Pulse modulation.** A thruster can't fire for less than 10 ms, but small corrections need far less per tick. Each thruster accumulates its requested firing time and fires once 10 ms has built up, so a 1% request becomes a 10 ms pulse every second.

## Detecting a stuck thruster

The navigation filter tracks only the target, so it cannot tell that the chaser's own thruster misbehaves. M14 adds a navigation-grade IMU and compares what it measures with what the commands should have produced.

- **A stuck-on thruster** adds its full push beyond whatever it was commanded. **A stuck-off one** removes what it was commanded. Each of the 32 possibilities (16 thrusters, two ways to fail) predicts a specific pattern in the IMU residual, tick by tick.
- **The test** is a generalized likelihood ratio for each possibility against no fault, over the last second. The likeliest is declared once its statistic passes 50. That thruster's valve closes, and the allocator stops using it.
- **When the fault started matters.** Two thrusters in the same pod push in exactly opposite directions. The first version assumed a fault had been there all window, and it blamed a stuck-on thruster on its opposite stuck off, whose pattern only begins when the controller starts fighting the disturbance. Estimating the onset as well fixed it.
- **A stuck-off thruster is invisible until commanded,** and harmless until then too. The suite times its detection from its first command after the failure.
- **Monitoring stops at capture.** The contact and the joined stack's mass break the free-flying model the residual relies on.

## Results

The suite flies M13's 28 docking episodes on thrusters, then 12 of them again with one failure each: stuck on or stuck off, a closing or a sideways thruster, triggered 12 m out, at the 5 m hold, or 1 m from contact.

- **All 40 docked.** At contact the chaser was a median 0.5 cm off the port's axis (worst 1.25 cm) and 0.07° off its alignment (worst 0.11°), against a 5 cm and 3° capture envelope.
- **Every failure was caught and isolated 0.1 s after it began,** or after the stuck-off thruster was first commanded, and it was always the right thruster and the right kind of failure. Each fault episode still docked.
- **No false alarms:** across the 28 healthy episodes nothing was isolated, and no failure hypothesis ever looked more likely than "no failure".
- **The thrusters could always do what was asked,** even after losing one.
- **A docking used 27–146 g of propellant** (median 62 g), more the faster the target tumbles: keeping station on a rotating docking axis takes constant thrust.
- **Pulsing leaves a small wobble:** the relative rotation at contact rose from essentially zero with M13's ideal actuators to about 0.01°/s, still a twentieth of the limit.

## Limitations

- The IMU feeds fault detection only; the chaser's own state is still exact truth for control.
- One thruster fails per episode, stuck fully on or fully off. Partial thrust loss, leaks, and two failures at once are not modeled.
- Mass is constant; the propellant used is reported, not subtracted.
- Thrusters fire on the 0.1 s tick, as impulse-equivalent averages; there is no plume impingement on the target and no valve dynamics.
- Fault detection stops at capture, so a failure while detumbling the stack goes unnoticed.

## Reproduce

```powershell
.\.venv\Scripts\rpod.exe m14-suite --output runs/m14 --label m14
```

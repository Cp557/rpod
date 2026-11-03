"""Propagate rpod cross-check cases with Basilisk: python propagate.py CASES.json OUTPUT.json"""

import json
from importlib.metadata import version
import sys

import numpy as np
from Basilisk.simulation import extForceTorque, spacecraft, svIntegrators
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody


def quaternion_mrp(quaternion_xyzw):
    """Modified Rodrigues parameters of a scalar-last quaternion, short rotation."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64) / np.linalg.norm(quaternion_xyzw)
    sign = 1.0 if w >= 0 else -1.0
    return (sign * np.array([x, y, z]) / (1 + sign * w)).tolist()


def propagate(case):
    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess("process")
    process.addTask(simulation.CreateNewTask("task", macros.sec2nano(case["dt_s"])))
    craft = spacecraft.Spacecraft()
    craft.hub.mHub = case["mass_kg"]
    craft.hub.IHubPntBc_B = np.diag(case["inertia_kg_m2"]).tolist()
    craft.hub.r_CN_NInit = case["position_m"]
    craft.hub.v_CN_NInit = case["velocity_mps"]
    craft.hub.sigma_BNInit = quaternion_mrp(case["quaternion_xyzw"])
    craft.hub.omega_BN_BInit = case["body_rate_radps"]
    integrator = svIntegrators.svIntegratorRK4(craft)
    craft.setIntegrator(integrator)
    simulation.AddModelToTask("task", craft)
    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True
    earth.mu = case["mu_m3_s2"]
    gravity.addBodiesTo(craft)
    wrench = extForceTorque.ExtForceTorque()
    wrench.extForce_B = case["force_body_n"]
    wrench.extTorquePntB_B = case["torque_body_nm"]
    craft.addDynamicEffector(wrench)
    simulation.AddModelToTask("task", wrench)
    recorder = craft.scStateOutMsg.recorder(macros.sec2nano(case["sample_s"]))
    simulation.AddModelToTask("task", recorder)
    simulation.InitializeSimulation()
    simulation.ConfigureStopTime(macros.sec2nano(case["duration_s"]))
    simulation.ExecuteSimulation()
    return {"times_s": (recorder.times() * 1e-9).tolist(),
            "positions_m": np.asarray(recorder.r_BN_N).tolist(),
            "velocities_mps": np.asarray(recorder.v_BN_N).tolist(),
            "mrp": np.asarray(recorder.sigma_BN).tolist(),
            "body_rates": np.asarray(recorder.omega_BN_B).tolist()}


def main(cases_path, output_path):
    with open(cases_path, encoding="utf-8") as stream:
        cases = json.load(stream)["cases"]
    result = {"software": {"basilisk": version("bsk"), "python": sys.version},
              "cases": {case["name"]: propagate(case) for case in cases}}
    with open(output_path, "x", encoding="utf-8") as stream:
        json.dump(result, stream)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Earth's shadow on the M15 orbit: the conical model against the cylindrical shadow."""

import unittest

import numpy as np

from rpod.rendezvous import SUN_ANGULAR_RADIUS_RAD, sunlight_direction, sunlit_fraction
from rpod.scenarios import EARTH_RADIUS_M, MU_M3_S2, ORBIT_RADIUS_M


class Shadow(unittest.TestCase):
    def test_full_light_umbra_and_the_edge_between(self):
        sunlight = sunlight_direction(0.0, 0.0)  # the Sun toward +X
        np.testing.assert_allclose(sunlight, [-1, 0, 0], atol=1e-15)
        self.assertEqual(sunlit_fraction([ORBIT_RADIUS_M, 0, 0], sunlight), 1.0)
        self.assertEqual(sunlit_fraction([-ORBIT_RADIUS_M, 0, 0], sunlight), 0.0)
        # Where the Sun's centre grazes Earth, the edge of the cylindrical shadow, about half
        # of its disk shows; across the edge the visible part falls monotonically.
        behind = np.sqrt(ORBIT_RADIUS_M**2 - EARTH_RADIUS_M**2)
        self.assertAlmostEqual(float(sunlit_fraction([-behind, EARTH_RADIUS_M, 0], sunlight)),
                               0.5, delta=0.01)
        sweep = np.arccos(behind / ORBIT_RADIUS_M) + np.linspace(-2, 2, 41) * SUN_ANGULAR_RADIUS_RAD
        positions = ORBIT_RADIUS_M * np.stack(
            (-np.cos(sweep), np.sin(sweep), np.zeros_like(sweep)), axis=-1)
        fraction = sunlit_fraction(positions, sunlight)
        self.assertEqual((fraction[0], fraction[-1]), (0.0, 1.0))
        self.assertTrue(np.all(np.diff(fraction) >= 0))
        # Where the disks just touch, the formula's terms cancel badly; it once gave 1.00003.
        limb = np.arcsin(EARTH_RADIUS_M / ORBIT_RADIUS_M)
        for edge in (1.0, -1.0):
            angle = limb + edge * SUN_ANGULAR_RADIUS_RAD
            touching = sunlit_fraction([ORBIT_RADIUS_M, 0, 0], [np.cos(angle), np.sin(angle), 0])
            self.assertTrue(0.0 <= touching <= 1.0)

    def test_shadow_time_per_orbit_matches_the_cylinder(self):
        rate = np.sqrt(MU_M3_S2 / ORBIT_RADIUS_M**3)
        samples = 100_000
        angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        orbit = ORBIT_RADIUS_M * np.stack((np.cos(angles), np.sin(angles), np.zeros(samples)), -1)
        step_s = 2 * np.pi / rate / samples
        for beta in (0.0, 30.0, 60.0):
            fraction = sunlit_fraction(orbit, sunlight_direction(beta, 40.0))
            cosine = np.sqrt(1 - (EARTH_RADIUS_M / ORBIT_RADIUS_M)**2) / np.cos(np.radians(beta))
            cylinder_s = 2 * np.arccos(cosine) / rate
            self.assertAlmostEqual((fraction < 0.5).sum() * step_s, cylinder_s, delta=0.5)
            self.assertLess((fraction == 0).sum() * step_s, cylinder_s)
        self.assertTrue(np.all(sunlit_fraction(orbit, sunlight_direction(75.0, 0.0)) == 1))


if __name__ == "__main__":
    unittest.main()

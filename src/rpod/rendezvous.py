"""Far-range rendezvous (M15): the Sun over a multi-orbit episode, and Earth's shadow."""

import numpy as np

from .scenarios import EARTH_RADIUS_M

# The realistic renderer draws the Sun as a uniform disk 0.53 deg across (blender_worker.py), so
# Earth's shadow there has a penumbra. This model uses the same disk.
SUN_ANGULAR_RADIUS_RAD = np.radians(0.53) / 2


def sunlight_direction(beta_deg, longitude_deg):
    """Inertial Sun incident-ray direction for the equatorial M15 orbit, whose normal is +Z.

    The Sun stands `beta_deg` above the orbit plane, at `longitude_deg` from +X within it.
    """
    beta, longitude = np.radians(beta_deg), np.radians(longitude_deg)
    return -np.array([np.cos(beta) * np.cos(longitude), np.cos(beta) * np.sin(longitude),
                      np.sin(beta)])


def sunlit_fraction(positions_m, sunlight_n, earth_radius_m=EARTH_RADIUS_M):
    """Fraction of the Sun's disk seen past a spherical Earth from each position (..., 3).

    1 in full sunlight, 0 in umbra. This is the conical shadow model of Montenbruck and Gill,
    Satellite Orbits (2000), section 3.4.2, with the Sun at infinity: `a` and `b` are the
    apparent radii of the Sun and Earth, and `c` the angle between their centres. Its terms
    cancel badly where the disks just touch, so the result is clipped to [0, 1].
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    toward_sun = -np.asarray(sunlight_n, dtype=np.float64)
    toward_sun = toward_sun / np.linalg.norm(toward_sun)
    distance = np.linalg.norm(positions, axis=-1)
    a = SUN_ANGULAR_RADIUS_RAD
    b = np.arcsin(earth_radius_m / distance)
    c = np.arccos(np.clip(-(positions @ toward_sun) / distance, -1.0, 1.0))
    x = (c**2 + a**2 - b**2) / (2 * np.maximum(c, 1e-12))
    y = np.sqrt(np.clip(a**2 - x**2, 0.0, None))
    hidden = (a**2 * np.arccos(np.clip(x / a, -1.0, 1.0))
              + b**2 * np.arccos(np.clip((c - x) / b, -1.0, 1.0)) - c * y)
    partial = np.clip(1 - hidden / (np.pi * a**2), 0.0, 1.0)
    return np.where(c >= a + b, 1.0, np.where(c <= b - a, 0.0, partial))

"""Blender process for the M11 showcase movie: the showcase chaser, plumes and filmic
settings around the unmarked target with stowed wings. Run with Blender's bundled Python.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_worker  # noqa: E402
from m11_worker import UnmarkedRenderer  # noqa: E402
from showcase_worker import ShowcaseRenderer  # noqa: E402


class UnmarkedShowcaseRenderer(ShowcaseRenderer, UnmarkedRenderer):
    """`ShowcaseRenderer` initializes over `UnmarkedRenderer`, which removes the cooperative
    aids and keeps the wings stowed."""


if __name__ == "__main__":
    # blender_worker.main constructs its module-level SceneRenderer.
    blender_worker.SceneRenderer = UnmarkedShowcaseRenderer
    blender_worker.main()

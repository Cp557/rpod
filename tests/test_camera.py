"""Camera frame/calibration contracts and image-only marked-target pose checks."""

import unittest

import cv2
import numpy as np

from rpod.approach import make_approach_scenario
from rpod.camera import (CameraModel, MARKER_CENTERS, camera_pose, earth_pose,
                         marker_corners, project_points, relative_target_pose)
from rpod.observations import ImageObservation, OwnNavigation
from rpod.perception import estimate_pose


def marker_image(rotation, translation, *, unknown=False):
    """Rasterize real marker pixels by plane homographies, without estimator inputs."""
    camera = CameraModel()
    image = np.full((camera.height, camera.width), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    source = np.float32([[0, 0], [599, 0], [599, 599], [0, 599]])
    for marker_id in MARKER_CENTERS:
        texture = cv2.aruco.generateImageMarker(dictionary, marker_id + (20 if unknown else 0), 600)
        destination = project_points(marker_corners(marker_id), rotation, translation, camera).astype(np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(texture, transform, (camera.width, camera.height), borderValue=255)
        image = np.minimum(image, warped)
    return image


class CameraGeometry(unittest.TestCase):
    def test_fixed_mount_and_optical_axes(self):
        camera = CameraModel()
        own = OwnNavigation(0, np.array([100.0, 200.0, 300.0]), np.zeros(3),
                            np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
        position, rotation = camera_pose(own, camera)
        np.testing.assert_allclose(position, [100.2, 200.55, 300.2])
        np.testing.assert_allclose(rotation @ [0, 0, 1], [0, 1, 0])
        np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, -1])
        target_position = own.position_m + [0, 11, 0]
        rotation_ct, translation_ct = relative_target_pose(own, target_position, own.quaternion_xyzw, camera)
        np.testing.assert_allclose(translation_ct, [-0.2, 0.2, 10.45], atol=1e-12)
        pixels = project_points(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0.1]]), rotation_ct, translation_ct, camera)
        self.assertGreater(pixels[1, 0], pixels[0, 0])
        self.assertLess(pixels[2, 1], pixels[0, 1])

    def test_projection_pixel_centers_and_invalid_geometry(self):
        camera = CameraModel()
        points = np.array([[0, 0, 0], [0.1, 0.2, 0]])
        projected = project_points(points, np.eye(3), [0, 0, 10], camera)
        np.testing.assert_allclose(projected[0], [511.5, 511.5])
        np.testing.assert_allclose(projected[1] - projected[0], camera.K[0, 0] * np.array([0.01, 0.02]))
        with self.assertRaisesRegex(ValueError, "in front"):
            project_points(points, np.eye(3), [0, 0, -10], camera)
        with self.assertRaises(ValueError):
            CameraModel(hfov_deg=0)
        centers = np.array([marker_corners(key).mean(axis=0) for key in MARKER_CENTERS])
        self.assertEqual(np.linalg.matrix_rank(centers - centers.mean(axis=0)), 3)

    def test_earth_stays_outside_the_view_on_the_along_track_approach(self):
        camera, scenario = CameraModel(), make_approach_scenario("approach")
        own = OwnNavigation(0, np.asarray(scenario["initial_positions_m"][1]), np.zeros(3),
                            np.asarray(scenario["initial_quaternions_xyzw"][1]), np.zeros(3))
        rotation_ce, center_c = earth_pose(own, camera)
        position_nc, rotation_nc = camera_pose(own, camera)
        np.testing.assert_allclose(rotation_ce, rotation_nc.T)
        np.testing.assert_allclose(rotation_nc @ center_c + position_nc, 0, atol=1e-6)
        # Earth's center is outside the image and no ray through the image border meets its
        # disk, so the convex disk misses the whole image.
        distance = np.linalg.norm(center_c)
        self.assertLess(center_c[2] / distance, 0.5)
        columns, rows = np.arange(camera.width), np.arange(camera.height)
        border = np.concatenate([
            np.c_[columns, np.zeros_like(columns)], np.c_[columns, np.full_like(columns, rows[-1])],
            np.c_[np.zeros_like(rows), rows], np.c_[np.full_like(rows, columns[-1]), rows],
        ])
        rays = np.c_[border, np.ones(len(border))] @ np.linalg.inv(camera.K).T
        cosines = rays @ center_c / np.linalg.norm(rays, axis=1) / distance
        self.assertLess(cosines.max(), np.sqrt(1 - (scenario["earth_radius_m"] / distance)**2))


class ImagePose(unittest.TestCase):
    def test_real_marker_pixels_recover_pose_and_preserve_timestamps(self):
        camera = CameraModel()
        rotation = np.asarray(camera.rotation_bc).T
        translation = np.array([-0.2, 0.2, 10.95])
        image = ImageObservation(17, 1_000_000_000, 1_200_000_000,
                                 marker_image(rotation, translation))
        estimate = estimate_pose(image, camera)
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual((estimate.capture_time_ns, estimate.delivery_time_ns),
                         (image.capture_time_ns, image.delivery_time_ns))
        self.assertEqual(estimate.marker_ids, (0, 1, 2, 3))
        self.assertGreaterEqual(estimate.inlier_count, 8)
        self.assertLess(np.linalg.norm(estimate.translation_ct - translation), 0.05)
        angle = np.arccos(np.clip((np.trace(estimate.rotation_ct.T @ rotation) - 1) / 2, -1, 1))
        self.assertLess(np.degrees(angle), 1.0)
        self.assertLess(estimate.reprojection_rms_px, 1.0)

    def test_missing_blank_and_unknown_marker_images_are_invalid(self):
        camera = CameraModel()
        rotation = np.asarray(camera.rotation_bc).T
        cases = (
            None,
            np.zeros((camera.height, camera.width), dtype=np.uint8),
            marker_image(rotation, np.array([-0.2, 0.2, 10.95]), unknown=True),
        )
        for pixels in cases:
            with self.subTest(missing=pixels is None):
                estimate = estimate_pose(ImageObservation(0, 0, 200_000_000, pixels), camera)
                self.assertFalse(estimate.valid)
                self.assertIsNone(estimate.rotation_ct)
                self.assertIsNone(estimate.translation_ct)


if __name__ == "__main__":
    unittest.main()

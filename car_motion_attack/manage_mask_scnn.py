from logging import getLogger

import cv2
import numpy as np
import pyopencl as cl
from car_motion_attack.utils import prg, queue, ctx
from car_motion_attack.utils import warp_corners, yuv2rgb, rgb2yuv
from car_motion_attack.config import (
    MODEL_IMG_HEIGHT,
    MODEL_IMG_WIDTH,
    IMG_CROP_HEIGHT,

    IMG_CROP_WIDTH,
    PIXELS_PER_METER,
)

from car_motion_attack.config import (
    SKY_HEIGHT,
    CAMERA_IMG_WIDTH,
    CAMERA_IMG_HEIGHT,
    #ROI_MAT,
    LATERAL_SHIFT_OFFSET,
    #ROI_MAT_INV,
)

MODEL_IMG_CH = 3

logger = getLogger(__name__)
mf = cl.mem_flags

LANE_INTERVAL = 93
LANE_WIDTH = 1.5
LANE_DOT_LENGTH = 23

class FrameMask:
    def __init__(
        self,
        global_mask,
        bev_mask,
        bev_corners,
        mtx_bev2camera,
        mtx_camera2bev,
        visible_patch_length,
        scale,
        left_lane_pos=4,
        right_lane_pos=36
    ):

        assert len(bev_mask.shape) == 2
        self.global_mask = global_mask
        self.bev_mask = bev_mask
        self.bev_corners = bev_corners
        self.visible_patch_length = visible_patch_length
        self.scale = scale

        self.left_lane_pos = left_lane_pos
        self.right_lane_pos = right_lane_pos

        self.mtx_bev2camera = mtx_bev2camera
        self.mtx_camera2bev = mtx_camera2bev

        self.patch_size = global_mask.shape

        self.lateral_shift = 0
        self.yaw_diff = 0
        self.longitudinal_shift = 0

        self.shifted_rotated_bev_mask = bev_mask
        self.shifted_rotated_bev_corners = bev_corners

        self.camera_mask = self._conv_mask_bev2camera()

        if self.bev_corners is not None:
            self.patch_corners = np.array(
                [
                    [0, 0],
                    [self.patch_size[1] * 2 - 1, 0],
                    [self.patch_size[1] * 2 - 1, self.visible_patch_length * 2 - 1],
                    [0, self.visible_patch_length * 2 - 1],
                ],
                dtype=np.float32,
            )

            self.bev_corners = self.bev_corners.astype(np.float32)
            self.camera_corners = warp_corners(self.mtx_bev2camera, self.bev_corners).astype(np.float32)

            self.mat_camera2patch = cv2.getPerspectiveTransform(self.camera_corners, self.patch_corners)
            self.mat_patch2camera = cv2.getPerspectiveTransform(self.patch_corners, self.camera_corners)

        else:
            self.bev_corners = None
            self.camera_corners = None

            self.mat_camera2patch = None
            self.mat_patch2camera = None

    def _conv_mask_bev2camera(self):
        logger.debug("enter")
        ret = cv2.warpPerspective(
            self.shifted_rotated_bev_mask,
            self.mtx_bev2camera,
            (CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT),
        )
        logger.debug("exit")
        return ret > 0.5# .astype(np.bool)

    def conv_camera2patch(self, mat_grad_rgb):
        if self.mat_camera2patch is None:
            return (
                np.ones(
                    shape=(
                        self.patch_size[0] * 2, self.patch_size[1] * 2,
                        MODEL_IMG_CH,
                    ),
                    dtype=mat_grad_rgb.dtype,
                )
                * np.nan
            )
        mat_grad_rgb = mat_grad_rgb[SKY_HEIGHT:]
        #mat_grad_rgb = cv2.GaussianBlur(mat_grad_rgb, (5, 5), 0)

        patch_rgb = cv2.warpPerspective(
            mat_grad_rgb,
            self.mat_camera2patch,
            (self.patch_size[1] * 2, self.patch_size[0] * 2),
            borderValue=np.nan,
        )
        #patch_rgb = cv2.GaussianBlur(patch_rgb, (5, 5), 0)

        # TODO
        #patch_rgb[~self.global_mask] = np.nan

        return patch_rgb

    def conv_patch2camera(self, patch_rgb):

        if self.mat_camera2patch is None:
            return (
                np.ones(
                    shape=(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH, MODEL_IMG_CH),
                    dtype=patch_rgb.dtype,
                )
                * np.nan
            )

        # TODO: draw lane lines
        #for i in range(0, patch_yuv.shape[0], LANE_INTERVAL * self.scale):
        #    if  self.left_lane_pos is not None:
        #        patch_yuv[i:i + LANE_DOT_LENGTH * self.scale, self.left_lane_pos * self.scale:int((self.left_lane_pos + LANE_WIDTH) * self.scale), :4] = 0.
        #    if self.right_lane_pos is not None:
        #        patch_yuv[i:i + LANE_DOT_LENGTH * self.scale, self.right_lane_pos * self.scale:int((self.right_lane_pos + LANE_WIDTH) * self.scale), :4] = 0.

        for i in range(0, patch_rgb.shape[0], LANE_INTERVAL * self.scale * 2):
            if  self.left_lane_pos is not None:
                patch_rgb[i:i + LANE_DOT_LENGTH * self.scale * 2, self.left_lane_pos * self.scale * 2:int((self.left_lane_pos + LANE_WIDTH) * self.scale * 2), :] = 127.
            if self.right_lane_pos is not None:
                patch_rgb[i:i + LANE_DOT_LENGTH * self.scale * 2, self.right_lane_pos * self.scale * 2:int((self.right_lane_pos + LANE_WIDTH) * self.scale * 2), :] = 127.

        visible_patch_rgb = patch_rgb[:self.visible_patch_length * 2]

        camera_rgb = cv2.warpPerspective(
            visible_patch_rgb,
            self.mat_patch2camera,
            (CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT),
            borderValue=(np.nan, np.nan, np.nan)
        )
        ## TODO 
        #camera_rgb = cv2.GaussianBlur(camera_rgb, (5, 5), 0)

        #camera_rgb[~self.camera_mask] = np.nan
        return camera_rgb


    def update_mask(self, lateral_shift, yaw_diff, longitudinal_shift=0):
        self.lateral_shift = lateral_shift
        self.yaw_diff = yaw_diff
        self.longitudinal_shift = longitudinal_shift

        if self.bev_corners is None:
            return

        lat_shift = self.lateral_shift * PIXELS_PER_METER * self.scale
        mtx_lateral_shift = np.float32([[1, 0, lat_shift], [0, 1, 0]])

        shifted_bev_mask = cv2.warpAffine(
            self.bev_mask,
            mtx_lateral_shift,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        lon_shift = self.longitudinal_shift * PIXELS_PER_METER * self.scale
        mtx_lon_shift = np.float32([[1, 0, 0], [0, 1, lon_shift]])

        shifted_bev_mask = cv2.warpAffine(
            shifted_bev_mask,
            mtx_lon_shift,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        cam_origin_shifted = (
            self.bev_mask.shape[1] // 2 - lat_shift + LATERAL_SHIFT_OFFSET,
            self.bev_mask.shape[0] - 1 - lon_shift,

        )

        mtx_rotation = cv2.getRotationMatrix2D(cam_origin_shifted, -yaw_diff, 1)

        self.shifted_rotated_bev_mask = cv2.warpAffine(
            shifted_bev_mask,
            mtx_rotation,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        shifted_corners = warp_corners(mtx_lateral_shift, self.bev_corners)
        shifted_corners = warp_corners(mtx_lon_shift, shifted_corners)
        self.shifted_rotated_bev_corners = warp_corners(mtx_rotation, shifted_corners)

        self.camera_mask = self._conv_mask_bev2camera()

        self.camera_corners = warp_corners(
            self.mtx_bev2camera, self.shifted_rotated_bev_corners
        ).astype(np.float32)

        self.mat_camera2patch = cv2.getPerspectiveTransform(
            self.camera_corners, self.patch_corners
        )
        self.mat_patch2camera = cv2.getPerspectiveTransform(
            self.patch_corners, self.camera_corners
        )

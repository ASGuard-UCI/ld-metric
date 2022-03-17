from logging import getLogger

import cv2
import sys
import numpy as np

from car_motion_attack.model_input_preprocess import ModelInPreprocess
from car_motion_attack.manage_mask_scnn import FrameMask
from car_motion_attack.perspective_transform import PerspectiveTransform
from car_motion_attack.model_output_postprocess import postprocess

from car_motion_attack.polyfuzz.polyfuzz import PolyFuzz, VehicleState
from car_motion_attack.polyfuzz.utils.parse_model_output import parse_model_output
from car_motion_attack.utils import warp_corners, yuv2rgb
from car_motion_attack.config import (
    CAMERA_IMG_HEIGHT,
    CAMERA_IMG_WIDTH,
    PIXELS_PER_METER,
    MODEL_IMG_HEIGHT,
    MODEL_IMG_WIDTH,
)
from car_motion_attack.config import (
    SKY_HEIGHT,
    PLAN_PER_PREDICT,
    IMG_CROP_HEIGHT,
    IMG_CROP_WIDTH,
    BEV_BASE_HEIGHT,
    BEV_BASE_WIDTH,
    INTRINSIC_MAT,
)

logger = getLogger(__name__)


def get_tran_mats(ext_mat, scale):
    inv_mat = np.linalg.inv(np.dot(INTRINSIC_MAT, ext_mat))
    inv_mat /= inv_mat[2, 2]

    m = np.dot(inv_mat, [[1, 0, 0], 
                    [0, 1, SKY_HEIGHT], 
                    [0, 0, 1]])

    d = PIXELS_PER_METER * scale
    m = np.dot([[0.7 * d, 0, 0], 
                [0, - d, 0], 
                [0, 0, 1]], m)
    m = np.dot([[0, 1, 0], 
                [-1, 0, 0], 
                [0, 0, 1]], m)

    m = np.dot([[1, 0, BEV_BASE_WIDTH * scale / 2], 
                [0, 1, BEV_BASE_HEIGHT * scale], 
                [0, 0, 1]], m)
    m /= m[2, 2]
    
    return m

class E2EController:

    def __init__(self):
        #self.model = SteeringModel(100)
        model_type = sys.argv[1]
        #self.model.load_state_dict(torch.load(f'weights/model_{model_type}_best.pth'))
        #self.model.load_state_dict(torch.load(f'weights/model_best.pth'))
        #self.model.eval()

    @staticmethod
    def get_steer_angle_op(PF, model_output, v=29, steering_angle=4):
        # PF = PolyFuzz()
        PF.update_state(v, steering_angle)
        path_poly, left_poly, right_poly, left_prob, right_prob = postprocess(model_output)

        valid, cost, angle = PF.run(left_poly, right_poly, path_poly, left_prob, right_prob)
        angle_steers_des = PF.angle_steers_des_mpc
        #angle_steers_des = 20
        return valid, cost, angle, angle_steers_des

class CarMotion:
    def __init__(self,
                 list_bgr_imgs,
                 df_sensors,
                 global_bev_mask,
                 roi_mat,
                 scale=1,
                 src_corners=None,
                 left_lane_pos=4,
                 right_lane_pos=36,
                 ext_mat=None
                 ):

        self.list_bgr_imgs = list_bgr_imgs

        self.df_sensors = df_sensors
        self.scale = scale
        self.n_frames = len(list_bgr_imgs)
        self.left_lane_pos = left_lane_pos
        self.right_lane_pos = right_lane_pos


        self.roi_mat = roi_mat
        self.roi_mat_inv = np.linalg.inv(roi_mat) if roi_mat is not None else None

        np.random.seed(0)
        self.global_bev_mask = global_bev_mask  # np.random.random((600, 33)) > 0
        self.global_patch_size = self.global_bev_mask.shape

        self.bev_img_height = (BEV_BASE_HEIGHT) * self.scale
        self.bev_img_width = BEV_BASE_WIDTH * self.scale

        if ext_mat is None:
            if src_corners is not None:
                self.src_corners = np.array(src_corners, dtype=np.float32)
            else:
                self.src_corners = np.array(
                    [[550.0, 400.0], [672.0, 400.0], [150.0, 600.0], [1150.0, 600.0]],
                    dtype=np.float32,
                )
            self.src_corners[:, 1] -= SKY_HEIGHT

            self.dist_corners = np.float32(
                [
                    [
                        (BEV_BASE_WIDTH * scale) // 2 - 25 * scale,
                        (BEV_BASE_HEIGHT) * scale * 0.665,
                    ],
                    [
                        (BEV_BASE_WIDTH * scale) // 2 + 25 * scale,
                        (BEV_BASE_HEIGHT) * scale * 0.665,
                    ],
                    [
                        (BEV_BASE_WIDTH * scale) // 2 - 25 * scale,
                        BEV_BASE_HEIGHT * scale * 0.99,
                    ],
                    [
                        (BEV_BASE_WIDTH * scale) // 2 + 25 * scale,
                        BEV_BASE_HEIGHT * scale * 0.99,
                    ],
                ]
            )

            self.mtx_camera2bev = cv2.getPerspectiveTransform(
                self.src_corners, self.dist_corners
            )
            self.mtx_bev2camera = cv2.getPerspectiveTransform(
                self.dist_corners, self.src_corners
            )
        else:
            self.mtx_camera2bev = get_tran_mats(ext_mat, scale)
            self.mtx_bev2camera = np.linalg.inv(self.mtx_camera2bev)
            self.mtx_bev2camera /= self.mtx_bev2camera[2, 2]

        #self.model_preprocess = ModelInPreprocess(self.roi_mat)

        self.list_transform = None  # PerspectiveTransform
        self.list_frame_mask = None  # FrameMask
        self.vehicle_state = None

        self.e2e_con = E2EController()

        self.total_lateral_shift = 0

    def setup_masks(self, starting_meters=90, lateral_shift=9):
        logger.debug("enter")

        self.list_transform = []
        self.list_frame_mask = []

        center = self.bev_img_width // 2 + lateral_shift * self.scale
        current_point = (
            self.bev_img_height - PIXELS_PER_METER * starting_meters * self.scale
        )

        for i in range(self.n_frames):
            current_point += (
                PIXELS_PER_METER * self.scale * self.df_sensors.loc[i, "distance"]
            )
            bev_mask = np.zeros(
                (self.bev_img_height, self.bev_img_width), dtype=np.float32
            )

            h_lo = int(current_point) - self.global_bev_mask.shape[0]
            h_hi = int(current_point)
            w_lo = center - self.global_bev_mask.shape[1] // 2
            w_hi = center + self.global_bev_mask.shape[1] // 2

            if w_hi - w_lo != self.global_bev_mask.shape[1]:
                w_hi += 1

            h_lo, h_hi = np.clip([h_lo, h_hi], 0, int(self.bev_img_height - 27 * self.scale))
            w_lo, w_hi = np.clip([w_lo, w_hi], 0, self.bev_img_width)

            visible_patch_length = h_hi - h_lo
            if visible_patch_length > 0:
                bev_mask[h_lo:h_hi, w_lo:w_hi] = self.global_bev_mask[: h_hi - h_lo, : w_hi - w_lo]
                bev_corners = np.array(
                    [[w_lo, h_lo], [w_hi, h_lo], [w_hi, h_hi], [w_lo, h_hi]]
                )
            else:
                bev_corners = None

            frame_mask = FrameMask(
                self.global_bev_mask,
                bev_mask,
                bev_corners,
                self.mtx_bev2camera,
                self.mtx_camera2bev,
                visible_patch_length,
                self.scale,
                self.left_lane_pos,
                self.right_lane_pos
            )
            self.list_frame_mask.append(frame_mask)

            p_transform = PerspectiveTransform(
                self.list_bgr_imgs[i],
                self.mtx_camera2bev,
                self.mtx_bev2camera,
                self.scale,
            )
            self.list_transform.append(p_transform)

            """
            ### debug
            tmp = self.list_bgr_imgs[i][SKY_HEIGHT:]
            mmm = np.zeros(shape=tmp.shape)
            mmm[IMG_CENTER_HEIGHT - 100 - SKY_HEIGHT: IMG_CENTER_HEIGHT + 100 - SKY_HEIGHT,
                IMG_CENTER_WIDTH - 200: IMG_CENTER_WIDTH + 200] = 1
            plt.imshow(tmp)
            plt.imshow(mmm, alpha=0.1)

            if camera_mask_corners is not None:
                plt.imshow(camera_mask, alpha=0.1)
                plt.scatter(camera_mask_corners[:, 0], camera_mask_corners[:, 1])
            plt.show()

            if camera_mask_corners is not None:
                plt.imshow(model_mask, alpha=0.1)
                plt.scatter(model_mask_corners[:, 0], model_mask_corners[:, 1])
            plt.show()
            ### debug
            """
        logger.debug("exit")

    def update_trajectory(
        self, model_outs, max_steering_angle_increase=0.25, start_steering_angle=None, add_noise=False
    ):
        logger.debug("enter")
        vehicle_state = VehicleState()

        if start_steering_angle is None:
            current_steering_angle = 0
        else:
            current_steering_angle = start_steering_angle

        vehicle_state.update_steer(current_steering_angle)

        self.total_lateral_shift = 0
        lateral_shift = 0
        lateral_shift_openpilot = 0

        yaw = 0
        yaw_diff = 0
        long_noise = 0
        logger.info("current_steering_angle: {}".format(current_steering_angle))

        list_total_lateral_shift = []
        list_lateral_shift_openpilot = []
        list_yaw = []
        list_desired_steering_angle = []
        list_current_steering_angle = []
        list_state = []
        ###

        PF = PolyFuzz()
        for i in range(self.n_frames):  # loop on 20Hz
            # update camera perspective
            if i > 0:
                yaw_diff = yaw - self.df_sensors.loc[i, "yaw"]
                lateral_shift = (
                    self.total_lateral_shift - self.df_sensors.loc[i, "lateral_shift"]
                )
                if add_noise:
                    yaw_diff += np.random.normal(0, abs(yaw_diff * 1.0e-2))
                    lateral_shift += np.random.normal(0, abs(lateral_shift * 1.0e-2))

                if "lateral_shift_openpilot" in self.df_sensors:
                    lateral_shift_openpilot = (
                        self.total_lateral_shift
                        - self.df_sensors.loc[i, "lateral_shift_openpilot"]
                    )
                else:
                    lateral_shift_openpilot = lateral_shift

                if add_noise:
                    long_noise = np.random.normal(0, 0.2)
                    self.list_transform[i].update_perspective(lateral_shift, yaw_diff, long_noise)
                else:
                    self.list_transform[i].update_perspective(lateral_shift, yaw_diff)
                logger.info(
                    "{}: yaw_diff: {}, lateral_shift: {}, total_lateral_shift: {}, lateral_shift_openpilot: {}".format(
                        i,
                        yaw_diff,
                        lateral_shift,
                        self.total_lateral_shift,
                        lateral_shift_openpilot,
                    )
                )
                logger.info(
                    "{}: current_yaw: {}, observed_yaw: {}, current_steering_angle: {}".format(
                        i, yaw, self.df_sensors.loc[i, "yaw"], current_steering_angle
                    )
                )

            # update patch
            if add_noise:
                self.list_frame_mask[i].update_mask(lateral_shift, yaw_diff, long_noise)
            else:
                self.list_frame_mask[i].update_mask(lateral_shift, yaw_diff)

            # update vehicle state
            v_ego = self.df_sensors.loc[i, "speed"]
            vehicle_state.update_velocity(v_ego)
            vehicle_state.update_steer(current_steering_angle)

            model_out = model_outs[i]
            try:
                valid, cost, angle, angle_steers_des_mpc = self.e2e_con.get_steer_angle_op(
                    PF, model_out, v_ego, current_steering_angle
                )
                if np.isnan(angle_steers_des_mpc):
                    raise Exception('fail motion model')
            except Exception as e:
                valid, cost, angle, angle_steers_des_mpc = True, 0, 0, current_steering_angle

            if i == 0 and start_steering_angle is None:
                current_steering_angle = angle_steers_des_mpc

            logger.info(
                "{}: valid: {}, cost: {}, angle: {}".format(i, valid, cost, angle)
            )

            logger.info(
                "desired steering angle: {}, current steering angle: {}".format(
                    angle_steers_des_mpc, current_steering_angle
                )
            )
            
            # update steering angle
            budget_steering_angle = angle_steers_des_mpc - current_steering_angle
            for _ in range(PLAN_PER_PREDICT):  # loop on 100Hz
                logger.debug(f"current_steering_angle 100Hz: {current_steering_angle}")
                angle_change = np.clip(
                    budget_steering_angle,
                    -max_steering_angle_increase,
                    max_steering_angle_increase,
                )
                current_steering_angle += angle_change
                budget_steering_angle -= angle_change
                if angle_steers_des_mpc - current_steering_angle > 0:
                    budget_steering_angle = max(budget_steering_angle, 0)
                else:
                    budget_steering_angle = min(budget_steering_angle, 0)

                state = vehicle_state.apply_plan(current_steering_angle)

            self.total_lateral_shift = state.y

            yaw = state.yaw
            self.yaw = yaw

            list_state.append(state)
            list_yaw.append(yaw)
            list_total_lateral_shift.append(self.total_lateral_shift)
            list_lateral_shift_openpilot.append(lateral_shift_openpilot)

            list_desired_steering_angle.append(angle_steers_des_mpc)
            list_current_steering_angle.append(current_steering_angle)

        self.list_state = list_state
        self.list_yaw = list_yaw
        self.list_total_lateral_shift = list_total_lateral_shift
        self.list_desired_steering_angle = list_desired_steering_angle
        self.list_current_steering_angle = list_current_steering_angle
        self.list_lateral_shift_openpilot = list_lateral_shift_openpilot

        logger.debug("exit")

    def update_trajectory_gen(
        self, model_outs_gen, max_steering_angle_increase=0.25, start_steering_angle=None, add_noise=False, target_frame=np.inf
    ):
        logger.debug("enter")
        vehicle_state = VehicleState()

        if start_steering_angle is None:
            current_steering_angle = 0
        else:
            current_steering_angle = start_steering_angle

        vehicle_state.update_steer(current_steering_angle)

        self.total_lateral_shift = 0
        lateral_shift = 0
        lateral_shift_openpilot = 0

        self.yaw = 0
        yaw_diff = 0
        long_noise = 0
        logger.info("current_steering_angle: {}".format(current_steering_angle))

        list_total_lateral_shift = []
        list_lateral_shift_openpilot = []
        list_yaw = []
        list_desired_steering_angle = []
        list_current_steering_angle = []
        list_state = []
        ###

        PF = PolyFuzz()
        

        for i in range(self.n_frames):  # loop on 20Hz
            if i >= target_frame:
                max_steering_angle_increase = np.inf

            # update camera perspective
            if i > 0:
                yaw_diff = self.yaw - self.df_sensors.loc[i, "yaw"]
                lateral_shift = (
                    self.total_lateral_shift - self.df_sensors.loc[i, "lateral_shift"]
                )
                if add_noise:
                    yaw_diff += np.random.normal(0, abs(yaw_diff * 1.0e-2))
                    lateral_shift += np.random.normal(0, abs(lateral_shift * 1.0e-2))

                if "lateral_shift_openpilot" in self.df_sensors:
                    lateral_shift_openpilot = (
                        self.total_lateral_shift
                        - self.df_sensors.loc[i, "lateral_shift_openpilot"]
                    )
                else:
                    lateral_shift_openpilot = lateral_shift

                if add_noise:
                    long_noise = np.random.normal(0, 0.2)
                    self.list_transform[i].update_perspective(lateral_shift, yaw_diff, long_noise)
                else:
                    self.list_transform[i].update_perspective(lateral_shift, yaw_diff)
                logger.info(
                    "{}: yaw_diff: {}, lateral_shift: {}, total_lateral_shift: {}, lateral_shift_openpilot: {}".format(
                        i,
                        yaw_diff,
                        lateral_shift,
                        self.total_lateral_shift,
                        lateral_shift_openpilot,
                    )
                )
                logger.info(
                    "{}: current_yaw: {}, observed_yaw: {}, current_steering_angle: {}".format(
                        i, self.yaw, self.df_sensors.loc[i, "yaw"], current_steering_angle
                    )
                )

            # update patch
            if add_noise:
                self.list_frame_mask[i].update_mask(lateral_shift, yaw_diff, long_noise)
            else:
                self.list_frame_mask[i].update_mask(lateral_shift, yaw_diff)

            # update vehicle state
            v_ego = self.df_sensors.loc[i, "speed"]
            vehicle_state.update_velocity(v_ego)
            vehicle_state.update_steer(current_steering_angle)

            model_out = next(model_outs_gen)
            try:
                valid, cost, angle, angle_steers_des_mpc = self.e2e_con.get_steer_angle_op(
                    PF, model_out, v_ego, current_steering_angle
                )
                if np.isnan(angle_steers_des_mpc):
                    raise Exception('fail motion model')
            except Exception as e:
                valid, cost, angle, angle_steers_des_mpc = True, 0, 0, current_steering_angle


            if i == 0 and start_steering_angle is None:
                current_steering_angle = angle_steers_des_mpc

            logger.info(
                "{}: valid: {}, cost: {}, angle: {}".format(i, valid, cost, angle)
            )

            logger.info(
                "desired steering angle: {}, current steering angle: {}".format(
                    angle_steers_des_mpc, current_steering_angle
                )
            )
            
            # update steering angle
            budget_steering_angle = angle_steers_des_mpc - current_steering_angle
            for _ in range(PLAN_PER_PREDICT):  # loop on 100Hz
                logger.debug(f"current_steering_angle 100Hz: {current_steering_angle}")
                angle_change = np.clip(
                    budget_steering_angle,
                    -max_steering_angle_increase,
                    max_steering_angle_increase,
                )
                current_steering_angle += angle_change
                budget_steering_angle -= angle_change
                if angle_steers_des_mpc - current_steering_angle > 0:
                    budget_steering_angle = max(budget_steering_angle, 0)
                else:
                    budget_steering_angle = min(budget_steering_angle, 0)

                state = vehicle_state.apply_plan(current_steering_angle)

            self.total_lateral_shift = state.y

            self.yaw = state.yaw

            list_state.append(state)
            list_yaw.append(self.yaw)
            list_total_lateral_shift.append(self.total_lateral_shift)
            list_lateral_shift_openpilot.append(lateral_shift_openpilot)

            list_desired_steering_angle.append(angle_steers_des_mpc)
            list_current_steering_angle.append(current_steering_angle)

        self.list_state = list_state
        self.list_yaw = list_yaw
        self.list_total_lateral_shift = list_total_lateral_shift
        self.list_desired_steering_angle = list_desired_steering_angle
        self.list_current_steering_angle = list_current_steering_angle
        self.list_lateral_shift_openpilot = list_lateral_shift_openpilot

        logger.debug("exit")

    def apply_noise(
        self, long_noise=(0, 1), lat_noise=(0, 0.1)
    ):
        for i in range(self.n_frames):  # loop on 20Hz
            _long_noise = np.random.normal(*long_noise)
            _lat_noise = np.random.normal(*lat_noise)
            self.list_transform[i].update_perspective(_lat_noise, 0, _long_noise)

        logger.debug("exit")


    def calc_model_inputs(self):
        logger.debug("enter")
        # sky_area = np.zeros((SKY_HEIGHT, CAMERA_IMG_WIDTH, 3), dtype=np.uint8)

        list_trainsformed_camera_imgs = [
            np.vstack([p.get_sky_img(mergin=30), p.shifted_roatated_camera_image[30:]])
            # np.vstack([sky_area, p.shifted_roatated_camera_image])
            for p in self.list_transform
        ]

        model_inputs = np.vstack(
            [
                self.model_preprocess.rgb_to_modelin(img)
                for img in list_trainsformed_camera_imgs
            ]
        )
        """
        model_inputs_c = np.vstack(
            [
                self.model_preprocess_c.rgb_to_modelin(img)
                for img in list_trainsformed_camera_imgs
            ]
        )

        for i in range(self.n_frames):
            print(np.abs(model_inputs_c[i] - model_inputs[i]).max())
        import pickle
        with open('modelin_debug_c.pkl', 'wb') as f:
            pickle.dump(model_inputs_c, f, -1)
        with open('modelin_debug_p.pkl', 'wb') as f:
            pickle.dump(model_inputs, f, -1)
        import pdb;pdb.set_trace()
        """
        logger.debug("exit")
        return model_inputs

    def calc_model_inputs_each(self, i):
        logger.debug("enter")
        p = self.list_transform[i]
        model_input = np.vstack([p.get_sky_img(mergin=30), p.shifted_roatated_camera_image[30:]])
        logger.debug("exit")
        return model_input


    def calc_model_inputs_rgb(self):
        logger.debug("enter")
        # sky_area = np.zeros((SKY_HEIGHT, CAMERA_IMG_WIDTH, 3), dtype=np.uint8)

        list_trainsformed_camera_imgs = [
            np.vstack([p.get_sky_img(mergin=30), p.shifted_roatated_camera_image[30:]])
            # np.vstack([sky_area, p.shifted_roatated_camera_image])
            for p in self.list_transform
        ]
        return list_trainsformed_camera_imgs

    def calc_attacked_model_inputs_rgb(self, patch, margin=10, return_camera_patch=False):
        logger.debug("enter")

        list_camera_patch = self.conv_patch2camera(patch)

        list_attacked_input = []
        for i, p in enumerate(self.list_transform):
            img = np.vstack(
                [p.get_sky_img(mergin=margin), 
                 np.where(np.isnan(list_camera_patch[i]), 
                                   p.shifted_roatated_camera_image,
                                   list_camera_patch[i])[margin:]
                ])
            mask = img.sum(axis=2) < 20

            img[mask] = p.img_all[mask]
            list_attacked_input.append(img)
        if return_camera_patch:
            return list_attacked_input, list_camera_patch
        else:
            return list_attacked_input

    def calc_attacked_model_inputs_rgb_each(self, i, patch, margin=10):
        logger.debug("enter")

        p = self.list_transform[i]
        img_cam = self.list_frame_mask[i].conv_patch2camera(patch)
        img = np.vstack(
            [p.get_sky_img(mergin=margin), 
                np.where(np.isnan(img_cam), 
                                p.shifted_roatated_camera_image,
                                img_cam)[margin:]
            ])
        mask = img.sum(axis=2) < 20
        img[mask] = p.img_all[mask]

        return img



    def conv_bev_image(self, img):
        logger.debug("enter")
        non_zoom_img = cv2.warpPerspective(
            img,
            self.mtx_camera2bev,
            (BEV_BASE_WIDTH * self.scale, (BEV_BASE_HEIGHT) * self.scale),
        )
        # zoom_img = non_zoom_img[450 * self.scale, 575 * self.scale: 625 * self.scale, :]
        logger.debug("exit")
        return non_zoom_img
    """
    def conv_model2patch(self, gradients):
        logger.debug("enter")
        list_patch_grads = [
            self.list_frame_mask[i].conv_model2patch(gradients[i])
            for i in range(len(gradients))
        ]
        logger.debug("exit")
        return list_patch_grads

    def conv_patch2model(self, patch, global_base_color):
        logger.debug("enter")
        list_model_purtabation = [
            self.list_frame_mask[i].conv_patch2model(patch, global_base_color)
            for i in range(self.n_frames)
        ]
        logger.debug("exit")
        return list_model_purtabation
    """


    def conv_camera2patch(self, gradients):
        logger.debug("enter")
        list_patch_grads = [
            self.list_frame_mask[i].conv_camera2patch(gradients[i])
            for i in range(len(gradients))
        ]
        logger.debug("exit")
        return list_patch_grads

    def conv_patch2camera(self, patch):
        logger.debug("enter")
        list_model_purtabation = [
            self.list_frame_mask[i].conv_patch2camera(patch)
            for i in range(self.n_frames)
        ]
        logger.debug("exit")
        return list_model_purtabation

    # debug methods for camera perspective
    def get_all_camera_images(self):
        logger.debug("enter")
        ret = [p.shifted_roatated_camera_image for p in self.list_transform]
        logger.debug("exit")
        return ret

    def get_all_camera_masks(self):
        logger.debug("enter")
        ret = [m.camera_mask for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    def get_all_camera_mask_corners(self):
        logger.debug("enter")
        ret = [m.camera_corners for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    def get_all_trainsformed_camera_imgs(self):
        logger.debug("enter")
        list_trainsformed_camera_imgs = [
            np.vstack([p.get_sky_img(mergin=30), p.shifted_roatated_camera_image[30:]])
            # np.vstack([sky_area, p.shifted_roatated_camera_image])
            for p in self.list_transform
        ]
        logger.debug("exit")
        return list_trainsformed_camera_imgs

    # debug methods for model perspective
    def get_all_model_images(self):
        logger.debug("enter")
        ret = self.calc_model_inputs()
        logger.debug("exit")
        return ret

    def get_all_model_masks(self):
        logger.debug("enter")
        ret = [m.model_mask for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    def get_all_model_mask_corners(self):
        logger.debug("enter")
        ret = [m.model_corners for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    # debug methods for bev perspective
    def get_all_bev_images(self):
        logger.debug("enter")
        ret = [p.shifted_roatated_bev_image for p in self.list_transform]
        logger.debug("exit")
        return ret

    def get_all_bev_masks(self):
        logger.debug("enter")
        ret = [m.bev_mask for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    def get_all_bev_mask_corners(self):
        logger.debug("enter")
        ret = [m.bev_corners for m in self.list_frame_mask]
        logger.debug("exit")
        return ret

    # debug methods for bev perspective
    def get_all_bev_images_with_patch(self, img, is_yuv6ch):
        if is_yuv6ch:
            patch = yuv2rgb(img)
        else:
            patch = img
        list_bev_img = [p.bev_image for p in self.list_transform]
        list_bev_mask = self.get_all_bev_masks()

        for i in range(self.n_frames):

            list_bev_img[i][list_bev_mask[i].astype(np.bool)] = patch.reshape((-1, 3))[
                : int(list_bev_mask[i].sum())
            ]
        return list_bev_img

    def get_all_camera_images_with_patrch(self, yuv_patch_6ch, is_yuv6ch=True):
        list_bev_img = self.get_all_bev_images_with_patch(
            yuv_patch_6ch, is_yuv6ch=is_yuv6ch
        )
        ret = []
        for i in range(self.n_frames):
            p = self.list_transform[i]
            bev_img = p.create_shifted_roatated_bev_image(list_bev_img[i])
            camera_img = p.conv_camera_image(bev_img)
            ret.append(camera_img)
        return ret

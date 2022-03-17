import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

from car_motion_attack.model_scnn import SCNNOpenPilot
from car_motion_attack.model_ultrafast import UltraFastOpenPilot
from car_motion_attack.model_polylanenet import PolyLaneNetOpenPilot

from car_motion_attack.config import MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH
from car_motion_attack.car_motion import CarMotion
from car_motion_attack.load_sensor_data import load_sensor_data
from car_motion_attack.utils import AdamOpt, yuv2rgb, rgb2yuv

from car_motion_attack.replay_bicycle import ReplayBicycle
#from car_motion_attack.loss import compute_path_pinv, loss_func
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX
                                      )
N_PREDICTIONS = 192
logger = getLogger(None)

class CarMotionAttack:
    def __init__(
        self,
        list_bgr_img,
        df_sensors,
        global_bev_mask,
        base_color,
        roi_mat,
        n_epoch=10000,
        learning_rate_patch=1.0e-2,
        learning_rate_color=1.0e-3,
        scale=1,
        result_dir='./result/',
        perturbable_area_ratio=10,
        is_attack_to_rigth=True,
        left_lane_pos=4,
        right_lane_pos=36,
        src_corners=None,
        target_deviation=0.5,
        l2_weight=0.01,
        target='laneatt',
        ext_mat=None,
    ):

        self.list_bgr_img = list_bgr_img
        self.n_frames = len(list_bgr_img)
        self.df_sensors = df_sensors
        self.result_dir = result_dir
        self.perturbable_area_ratio = perturbable_area_ratio
        self.base_color = base_color
        self.roi_mat = roi_mat
        self.is_attack_to_rigth = is_attack_to_rigth
        self.left_lane_pos = left_lane_pos
        self.right_lane_pos = right_lane_pos
        self.target_deviation = target_deviation
        self.l2_weight = l2_weight
        self.scale = scale

        self.last_epoch = None

        self.global_bev_mask = global_bev_mask
        self.car_motion = CarMotion(
            self.list_bgr_img,
            self.df_sensors,
            self.global_bev_mask,
            self.roi_mat,
            left_lane_pos=left_lane_pos,
            right_lane_pos=right_lane_pos,
            scale=scale,
            src_corners=src_corners,
            ext_mat=ext_mat
        )

        self.global_bev_purtabation = (
            np.random.random(
                (self.global_bev_mask.shape[0], self.global_bev_mask.shape[1], 6),
            ).astype(DTYPE)
            * 1.0e-8
        )
        self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()

        self.global_base_color = np.array(
            [base_color, 0, 0], dtype=DTYPE
        )  # np.zeros(3, dtype=DTYPE)

        if target == 'scnn':
            self.model = SCNNOpenPilot(ext_mat, mtx_bev2camera=self.car_motion.mtx_bev2camera)
        elif target == 'laneatt':
            from car_motion_attack.model_laneatt import LaneATTOpenPilot
            self.model = LaneATTOpenPilot(ext_mat, mtx_bev2camera=self.car_motion.mtx_bev2camera)
        elif target == 'ultrafast':
            self.model = UltraFastOpenPilot(ext_mat, mtx_bev2camera=self.car_motion.mtx_bev2camera)
        elif target == 'polylanenet':
            self.model = PolyLaneNetOpenPilot(ext_mat, mtx_bev2camera=self.car_motion.mtx_bev2camera)
        elif target == 'nan':
            self.model = None
        else:
            raise Exception(f'Invalid target: {target}')

        self.n_epoch = n_epoch + 1
        self.learning_rate_patch = learning_rate_patch
        self.learning_rate_color = learning_rate_color

        #self._create_tf_variables()

    def run(
        self,
        lateral_shift=4,
        starting_meters=60,
        starting_steering_angle=True,
        starting_patch_dir=None,
        starting_patch_epoch=None,
        trajectory_update=True
    ):
        logger.debug("enter")
        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )

        if self.base_color is None:
            bev_img = self.car_motion.list_transform[0].bev_image
            bev_mask = self.car_motion.list_frame_mask[0].bev_mask

            self.base_color = rgb2yuv(np.array([[bev_img[bev_mask > 0].mean(axis=0).astype(int)]* 2] * 2))[0, 0, 0]
            self.global_base_color = np.array(
                        [self.base_color, 0, 0], dtype=DTYPE
                    )  # np.zeros(3, dtype=DTYPE)


        # ops

        if starting_patch_dir is not None:
            self.global_bev_purtabation = np.load(
                starting_patch_dir + f"_global_patch_{starting_patch_epoch}.npy"
            )
            self.masked_global_bev_purtabation = np.load(
                starting_patch_dir + f"_global_masked_patch_{starting_patch_epoch}.npy"
            )
            self.global_base_color = np.load(
                starting_patch_dir + f"_global_base_color_{starting_patch_epoch}.npy"
            )

        adam_patch = AdamOpt(
            yuv2rgb(self.global_bev_purtabation).shape, lr=self.learning_rate_patch
        )

        color_6ch = np.array([self.global_base_color[0]] * 4 + [self.global_base_color[1]] + [self.global_base_color[2]])

        #self.sess.run(
        #    self.ops_base_color_update,
        #    feed_dict={self.yuv_color: self.global_base_color},
        #)
        # optimization iteration
        for epoch in tqdm(range(self.n_epoch)):
        #for epoch in tqdm(range(80)):

            logger.debug("start {}".format(epoch))

            logger.debug("calc model ouput")
            #model_img_inputs = self.car_motion.calc_model_inputs_rgb()

            logger.debug("apply global purtabation to each frame")

            patch_yuv = self.masked_global_bev_purtabation + color_6ch

            patch_rgb = yuv2rgb(patch_yuv).clip(0, 255)

            #list_patches = self.car_motion.conv_patch2camera(patch_rgb)

            list_attacked_input = self.car_motion.calc_attacked_model_inputs_rgb(patch_rgb)

            logger.debug("update car trajectory")
            model_attack_outputs = []#np.vstack(self.sess.run(self.list_ops_predicts))
            #model_seg_pred = []
            model_output = np.ones(1760)
            if trajectory_update:
                for i in range(self.n_frames):
                    try:
                        model_output = self.model.predict(list_attacked_input[i])
                    except:
                        pass
                    model_attack_outputs.append(model_output)
                    #model_seg_pred.append(self.model.seg_pred)
                model_attack_outputs = np.vstack(model_attack_outputs)
            
                self.car_motion.update_trajectory(
                    model_attack_outputs, start_steering_angle=starting_steering_angle, add_noise=True
                )
            else:
                model_attack_outputs = None
                self.car_motion.apply_noise()
            ### for debug
            #with open(self.result_dir + 'model_attack_input.pkl', 'wb') as f:
            #    pickle.dump(list_attacked_input, f, -1)
            #with open(self.result_dir + 'model_attack_outputs.pkl', 'wb') as f:
            #    pickle.dump(model_attack_outputs, f, -1)
            #with open(self.result_dir + 'model_seg_pred.pkl', 'wb') as f:
            #    pickle.dump(model_seg_pred, f, -1)
            #if self.car_motion.list_desired_steering_angle[0] < -20:
            #    break
            #import pdb;pdb.set_trace()

            logger.debug("calc gradients")
            list_var_grad = [self.model.get_input_gradient(list_attacked_input[i])
                            for i in range(self.n_frames)]
            list_var_grad = np.stack(list_var_grad)

            #np.save('list_var_grad', list_var_grad)

            logger.debug("conv gradients -> patch")
            logger.debug("agg patch grads")
            patch_grad = self._agg_gradients(list_var_grad)

            patch_grad = np.sign(patch_grad) * 255
            
            #np.save('patch_grad', patch_grad)
            logger.debug("update global purtabation")

            patch_rgb = (patch_rgb - adam_patch.update(patch_grad / 255) * 255).clip(0, 255)
            adam_patch.lr *= 0.99
            #patch_rgb = (patch_rgb - patch_grad * self.learning_rate_patch).clip(0, 255)

            patch_yuv = rgb2yuv(patch_rgb)
            perturb_yuv = patch_yuv - color_6ch
            perturb_yuv -= self.learning_rate_patch * 2 * self.l2_weight * perturb_yuv

            self.global_bev_purtabation = perturb_yuv #np.where(np.isnan(perturb_yuv), self.global_bev_purtabation, perturb_yuv)

            self.global_bev_purtabation[:, :, 4:] = 0


            if (epoch) % min(10, self.n_epoch - 1) == 0:
                patch_diff = self.global_bev_purtabation.clip(0, None).sum(axis=2)
                patch_diff += np.random.random(patch_diff.shape) * 1.0e-8 # tie break
                threshold = np.percentile(patch_diff, 100 - self.perturbable_area_ratio)

                mask_bev_purtabation = patch_diff > threshold

                self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()
                self.masked_global_bev_purtabation[~mask_bev_purtabation] = 0.
            else:
                self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()
            self.masked_global_bev_purtabation = self.masked_global_bev_purtabation.clip(
                0, 0.83122042#- self.base_color
            )

            if (epoch) % min(50, self.n_epoch - 1) == 0 and epoch > 0:
                np.save(
                    self.result_dir + f"_global_patch_{epoch}",
                    self.global_bev_purtabation,
                )
                np.save(
                    self.result_dir + f"_global_masked_patch_{epoch}",
                    self.masked_global_bev_purtabation,
                )
                np.save(
                    self.result_dir + f"_global_base_color_{epoch}",
                    self.global_base_color,
                )
                #np.save(
                #    self.result_dir + f"model_img_inputs_{epoch}",
                #    np.stack(list_attacked_input),
                #)

                #model_imgs = np.vstack(self.sess.run(self.list_ops_model_img))
                np.save(
                    self.result_dir + f"model_outputs_{epoch}", model_attack_outputs
                )
                #np.save(self.result_dir + f"model_img_inputs_{epoch}", model_imgs)
                #logger.info(
                #    f"save epoch: {epoch + 1}, total_lat: {self.car_motion.list_total_lateral_shift} desired: {self.car_motion.list_desired_steering_angle}"
                #)

                if trajectory_update:
                    #if (
                    #    (self.is_attack_to_rigth and self.car_motion.list_lateral_shift_openpilot[-1] < - self.target_deviation) or
                    #    ((not self.is_attack_to_rigth)
                    #    and self.car_motion.list_lateral_shift_openpilot[-1] > self.target_deviation)
                    #):
                    if np.abs(self.car_motion.list_lateral_shift_openpilot).max() > self.target_deviation:
                        logger.info(
                            f"Reached target deviation: {epoch + 1}, total_lat: {self.car_motion.list_lateral_shift_openpilot[-1]}"
                        )
                        self.last_epoch = epoch
                        break

        self.last_epoch = epoch
        logger.debug("exit")


    def replay(
        self,
        epoch,
        lateral_shift=4,
        starting_meters=60,
        starting_steering_angle=None,
        trajectory_update=True
    ):
        logger.debug("enter")
        output_dir = self.result_dir + '/replay/'

        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )

        self.global_bev_purtabation = np.load(
            self.result_dir + f"_global_patch_{epoch}.npy"
        )
        self.masked_global_bev_purtabation = np.load(
            self.result_dir + f"_global_masked_patch_{epoch}.npy"
        )
        self.global_base_color = np.load(
            self.result_dir + f"_global_base_color_{epoch}.npy"
        )

        color_6ch = np.array([self.global_base_color[0]] * 4 + [self.global_base_color[1]] + [self.global_base_color[2]])


        patch_yuv = self.masked_global_bev_purtabation + color_6ch
        patch_rgb = yuv2rgb(patch_yuv).clip(0, 255)

        model_image_inputs = []
        model_outputs = []
        model_lane_pred = []
        def pred_generator():
            model_output = np.zeros(MODEL_OUTPUT_SHAPE[1:])
            for i in range(self.n_frames):
                patch_model = self.car_motion.calc_attacked_model_inputs_rgb_each(
                    i, patch_rgb
                )
                try:
                    model_output = self.model.predict(patch_model)
                except:
                    pass
                
                model_image_inputs.append(patch_model)
                model_outputs.append(model_output)
                model_lane_pred.append([self.model.left_line, self.model.right_line])
                yield model_output

        if trajectory_update:
            self.car_motion.update_trajectory_gen(
                pred_generator(), start_steering_angle=starting_steering_angle # with limit
            )
        else:
            for _ in pred_generator():
                pass
                
        ### for debug
        with open(output_dir + 'model_attack_input.pkl', 'wb') as f:
            pickle.dump(model_image_inputs, f, -1)
        with open(output_dir + 'model_attack_outputs.pkl', 'wb') as f:
            pickle.dump(model_outputs, f, -1)
        with open(output_dir + 'model_lane_pred.pkl', 'wb') as f:
            pickle.dump(model_lane_pred, f, -1)

        with open(output_dir + 'global_patch.pkl', 'wb') as f:
            pickle.dump(self.global_bev_purtabation, f, -1)
        with open(output_dir + 'global_masked_patch.pkl', 'wb') as f:
            pickle.dump(self.masked_global_bev_purtabation, f, -1)
        with open(output_dir + 'global_base_color.pkl', 'wb') as f:
            pickle.dump(self.global_base_color, f, -1)

        self.last_epoch = epoch
        logger.debug("exit")

    def calc_metric(
        self,
        epoch,
        target_frame,
        lateral_shift=4,
        starting_meters=60,
        starting_steering_angle=None,
        trajectory_update=True
    ):
        logger.debug("enter")
        output_dir = self.result_dir + '/replay/'

        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )

        self.global_bev_purtabation = np.load(
            self.result_dir + f"_global_patch_{epoch}.npy"
        )
        self.masked_global_bev_purtabation = np.load(
            self.result_dir + f"_global_masked_patch_{epoch}.npy"
        )
        self.global_base_color = np.load(
            self.result_dir + f"_global_base_color_{epoch}.npy"
        )

        color_6ch = np.array([self.global_base_color[0]] * 4 + [self.global_base_color[1]] + [self.global_base_color[2]])


        patch_yuv = self.masked_global_bev_purtabation + color_6ch
        patch_rgb = yuv2rgb(patch_yuv).clip(0, 255)

        model_image_inputs = []
        model_outputs = []
        model_lane_pred = []
        def pred_generator():
            model_output = np.zeros(MODEL_OUTPUT_SHAPE[1:])
            gt_pos = self.df_sensors[['longitude_shift', 'lateral_shift']].values
            for i in range(self.n_frames):
                if i <= target_frame:
                    patch_model = self.car_motion.calc_attacked_model_inputs_rgb_each(
                        i, patch_rgb
                    )
                    try:
                        model_output = self.model.predict(patch_model)
                    except:
                        pass
                    model_image_inputs.append(patch_model)
                else:
                    # centered vehicle
                    pos = np.array(gt_pos) - [gt_pos[i, 0], self.car_motion.total_lateral_shift]
                    # apply yaw
                    _cos = np.cos(self.car_motion.yaw)
                    _sin = - np.sin(self.car_motion.yaw)
                    mat_rotate = np.array([[_cos, -_sin], [_sin, _cos]])
                    pos = np.dot(pos, mat_rotate.T)
                    # interpolate
                    cs_path = InterpolatedUnivariateSpline(pos[:, 0], pos[:, 1], k=3, ext=3)
                    
                    # draw
                    # 0.0585227139336881
                    
                    x = np.arange(N_PREDICTIONS) - 12
                    y = cs_path(x) - 0.05# + (self.car_motion.toal_lateral_shift)# - lateral_pos[i])

                    path_start = 0
                    left_start = N_PREDICTIONS * 2
                    right_start = N_PREDICTIONS * 2 + N_PREDICTIONS * 2 + 1

                    model_output = np.ones(1760)

                    model_output[path_start:path_start + N_PREDICTIONS] = y
                    model_output[left_start:left_start + N_PREDICTIONS] = y
                    model_output[right_start:right_start + N_PREDICTIONS] = y

                model_outputs.append(model_output)
                model_lane_pred.append([self.model.left_line, self.model.right_line])
                yield model_output

        if trajectory_update:
            self.car_motion.update_trajectory_gen(
                pred_generator(), start_steering_angle=starting_steering_angle, target_frame=target_frame # with limit
            )
        else:
            for _ in pred_generator():
                pass
        self.last_epoch = epoch
        logger.debug("exit")


    def _agg_gradients(self, list_var_grad):
        """
        model_mask_areas = np.array(
            [m.sum() for m in self.car_motion.get_all_camera_masks()]
        )
        weights = model_mask_areas / model_mask_areas.sum()

        list_patch_grad = self.car_motion.conv_camera2patch(
            list_var_grad
        )  # zero is missing value
        for i in range(len(list_patch_grad)):
            list_patch_grad[i] *= weights[i]

        tmp = np.stack(list_patch_grad)
        
        tmp = np.nanmean(tmp, axis=0)
        tmp[np.isnan(tmp)] = 0
        """

        list_patch_grad = self.car_motion.conv_camera2patch(
            list_var_grad
        )  # zero is missing value

        tmp = np.nanmean(list_patch_grad, axis=0)
        tmp[np.isnan(tmp)] = 0
        return tmp

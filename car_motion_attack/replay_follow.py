import pickle
import os
import sys

from logging import getLogger
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import InterpolatedUnivariateSpline

#import tensorflow as tf
#from keras import backend as K
from tqdm import tqdm
from car_motion_attack.model_scnn import SCNNOpenPilot
from car_motion_attack.model_laneatt import LaneATTOpenPilot
from car_motion_attack.model_ultrafast import UltraFastOpenPilot
from car_motion_attack.model_polylanenet import PolyLaneNetOpenPilot

from car_motion_attack.load_sensor_data import load_sensor_data
from car_motion_attack.car_motion import CarMotion
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH
                                      )

N_PREDICTIONS = 192

logger = getLogger(None)


class ReplayMetric:
    def __init__(
        self,
        list_bgr_img,
        df_sensors,
        global_bev_mask,
        roi_mat,
        src_corners,
        n_epoch=1001,
        learning_rate_patch=1.0e-2,
        learning_rate_color=1.0e-3,
        scale=1,
        target='laneatt',
        ext_mat=None,
        lane_detect=None
    ):
        self.lane_detect = lane_detect

        self.list_bgr_img = list_bgr_img
        self.n_frames = len(list_bgr_img)
        self.df_sensors = df_sensors
        self.roi_mat = roi_mat
        self.src_corners = src_corners

        self.global_bev_mask = global_bev_mask
        self.car_motion = CarMotion(
            self.list_bgr_img,
            self.df_sensors,
            self.global_bev_mask,
            self.roi_mat,
            scale=scale,
            src_corners=src_corners,
            ext_mat=ext_mat
        )

        self.global_bev_purtabation = (
            np.ones(
                (self.global_bev_mask.shape[0], self.global_bev_mask.shape[1], 6),
                dtype=DTYPE,
            )
            * 1.0e-10
        )
        self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()

        self.global_base_color = np.array(
            [-0.7, 0, 0], dtype=DTYPE
        )  # np.zeros(3, dtype=DTYPE)

        if target == 'scnn':
            self.model = SCNNOpenPilot(ext_mat)
        elif target == 'laneatt':
            self.model = LaneATTOpenPilot(ext_mat)
        elif target == 'ultrafast':
            self.model = UltraFastOpenPilot(ext_mat)
        elif target == 'polylanenet':
            self.model = PolyLaneNetOpenPilot(ext_mat)
        else:
            raise Exception(f'Invalid target: {target}')
        self.n_epoch = n_epoch
        self.learning_rate_patch = learning_rate_patch
        self.learning_rate_color = learning_rate_color

    def run(self, lateral_shift=4, starting_meters=60, start_steering_angle=None, trajectory_update=True):
        logger.debug("enter")
        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )
        # self.list_ops_model_img = self.list_tf_model_imgs
        #model_rnn_inputs = []
        model_outputs = []
        model_inputs = []
        model_lane_pred = []

        def pred_generator():
            #rnn_input = np.zeros(RNN_INPUT_SHAPE)
            #desire_input = np.zeros(MODEL_DESIRE_INPUT_SHAPE)
            model_output = np.zeros(MODEL_OUTPUT_SHAPE[1:])

            #lateral_pos = self.df_sensors['lateral_shift'].values
            #long_pos = self.df_sensors['longitude_shift'].values

            gt_pos = self.df_sensors[['longitude_shift', 'lateral_shift']].values

            

            for i in range(self.n_frames):
                model_img_nopatch = self.car_motion.calc_model_inputs_each(i)#.reshape(IMG_INPUT_SHAPE)
                if 0:#i == 0:
                    
                    #try:
                    if self.lane_detect is None:
                        model_output = self.model.predict(model_img_nopatch)
                    else:
                        model_output = self.lane_detect[i]
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
                    x = np.arange(N_PREDICTIONS)
                    y = cs_path(x)# + (self.car_motion.total_lateral_shift)# - lateral_pos[i])

                    path_start = 0
                    left_start = N_PREDICTIONS * 2
                    right_start = N_PREDICTIONS * 2 + N_PREDICTIONS * 2 + 1

                    model_output = np.ones(1760)

                    model_output[path_start:path_start + N_PREDICTIONS] = y
                    model_output[left_start:left_start + N_PREDICTIONS] = y
                    model_output[right_start:right_start + N_PREDICTIONS] = y
                #except:
                #    pass
                model_lane_pred.append([self.model.left_line, self.model.right_line])
                model_outputs.append(model_output)
                model_inputs.append(model_img_nopatch)

                yield model_output
        if trajectory_update:
            self.car_motion.update_trajectory_gen(
                pred_generator(), start_steering_angle=start_steering_angle, max_steering_angle_increase=np.inf
            )
        else:
            for _ in pred_generator():
                pass
        self.model_lane_pred = model_lane_pred
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        logger.debug("exit")
        print(self.car_motion.list_total_lateral_shift)
        return self.car_motion

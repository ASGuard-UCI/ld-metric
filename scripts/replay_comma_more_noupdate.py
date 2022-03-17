#!/usr/bin/env python3
import sys
import pickle
import os
import gc
import json
from logging import getLogger


import numpy as np
import pandas as pd
import cv2

#import tensorflow as tf
from tqdm import tqdm

try:
    APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(APP_ROOT)
    sys.path.append('/openpilot')
    sys.path.append(APP_ROOT + '/LaneATT')
    sys.path.append(APP_ROOT + '/Ultra-Fast-Lane-Detection')
except:
    raise

from car_motion_attack.attack import CarMotionAttack
from car_motion_attack.replay_bicycle import ReplayBicycle
from car_motion_attack.load_sensor_data import load_sensor_data, load_transform_matrix
from car_motion_attack.load_sensor_data import load_transform_matrix


logger = getLogger(None)

model_type = sys.argv[1]

if model_type not in ('laneatt', 'ultrafast', 'scnn', 'polylanenet'):
    raise Exception(f'unknown model: {model_type}')

from tools.lib.logreader import LogReader
def get_ext_mat(path):
    lr = LogReader(path + 'raw_log.bz2')
    models = [l for l in lr if l.which() == 'liveCalibration']
    mats = [np.array(m.liveCalibration.extrinsicMatrix).reshape(3, 4) for m in models]
    mat = np.mean(mats, axis=0)
    mat = mat[:, [0, 1, 3]]
    return mat

def get_models(path):
    N_PREDICTIONS = 192
    path_start = 0
    left_start = N_PREDICTIONS * 2
    right_start = N_PREDICTIONS * 2 + N_PREDICTIONS * 2 + 1


    ret = []
    lr = LogReader(path + 'raw_log.bz2')
    m = [l for l in lr if l.which() == 'model']
    for l in m:

        output = np.ones(1760)
        ll = np.array(l.model.leftLane.points)
        rr = np.array(l.model.rightLane.points)
        output[path_start:path_start + 50] = np.array(l.model.path.points)
        output[left_start:left_start + 50] = ll - 1.8
        output[right_start:right_start + 50] = rr + 1.8
        ret.append(output)
    return ret

def run(data_path='',
         n_epoch=10000,
         n_frames=20,
         scale=5,
         base_color=0.38,
         starting_meters=45,
         patch_lateral_shift=0,
         result_dir='./result/',
         left_lane_pos=4,
         right_lane_pos=36,
         left_solid=False,
         right_solid=False,
         src_corners=None,
         target_deviation=0.5,
         is_attack_to_rigth=True,
         patch_width=45,
         patch_length=300,
         frame_offset=0,
         l2_weight=0.01
         ):
    if os.path.exists(data_path + 'raw_log_bz2'):
        os.rename(data_path + 'raw_log_bz2', data_path + 'raw_log.bz2')


    df_sensors = load_sensor_data(data_path, offset=frame_offset).head(n_frames + 1)

    #import pdb;pdb.set_trace()
    #roi_mat = load_transform_matrix(data_path + 'raw_log.bz2', start_time=df_sensors.loc[0, 't'])
    roi_mat = load_transform_matrix(data_path + 'raw_log.bz2', start_time=df_sensors.loc[0, 't'])
    np.save(data_path + 'trns', roi_mat)
    ext_mat = get_ext_mat(data_path)
    np.save(data_path + 'ext_mat', ext_mat)

    #lane_detect = get_models(data_path)
    list_bgr_img = []
    vc = cv2.VideoCapture(data_path + 'video.hevc')
    #vc.set(cv2.CAP_PROP_POS_FRAMES, frame_offset) does not work!
    for _ in range(frame_offset):
        vc.read()
    for i in range(n_frames + 1):
        rval, frame = vc.read()
        list_bgr_img.append(frame)

    global_bev_mask = np.random.random((patch_length * scale, patch_width * scale)) > 0

    _src_corners = np.array(src_corners)

    rb = ReplayBicycle(
        list_bgr_img, df_sensors, global_bev_mask, roi_mat, _src_corners, scale=scale, target=model_type, ext_mat=ext_mat,
        #lane_detect=lane_detect
    )
    cm = rb.run(start_steering_angle=None)

    with open(result_dir + '/e2e/model_benign_lane_pred.pkl', 'wb') as f:
        pickle.dump(rb.model_lane_pred, f, -1)
    with open(result_dir + '/e2e/model_benign_inputs.pkl', 'wb') as f:
        pickle.dump(rb.model_inputs, f, -1)
    with open(result_dir + '/e2e/model_benign_outputs.pkl', 'wb') as f:
        pickle.dump(rb.model_outputs, f, -1)

def main(row, offset):

    config = {}    
    config['result_dir'] = f'logs/comma_benign_noupdate/replay_{model_type}_more_noupdate_newloss/{row.sc_name}/'
    config['data_path'] = f'comma2k19-ld/{row.sc_name}/'

    config['n_frames'] = 20
    config['frame_offset'] = offset
    os.makedirs(config['result_dir'] + '/e2e/', exist_ok=True)

    for h in logger.handlers:
        logger.removeHandler(h)
    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    )

    logger.setLevel('INFO')


    handler = FileHandler(
        config['result_dir'] + os.path.basename(os.path.abspath(__file__)) + f'_f{offset}.log', 'w'
        )
    handler.setFormatter(log_fmt)
    handler.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('start')
    run(**config)
    logger.info('end')

if __name__ == '__main__':

    from logging import StreamHandler, Formatter, FileHandler

    df_sc = pd.read_csv('data/scenario_index.csv')
    for _, row in tqdm(list(df_sc.iterrows()), desc='sc_loop', ncols=55):
        #for offset in tqdm(range(20), ncols=55):
        main(row, 0)

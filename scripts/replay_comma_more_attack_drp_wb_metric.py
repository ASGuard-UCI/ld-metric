import sys
import pickle
import os
import gc
import json
from logging import getLogger


import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import InterpolatedUnivariateSpline

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

from car_motion_attack.config import SKY_HEIGHT

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
         l2_weight=0.01,
         target_frame=None,
         ):


    if os.path.exists(data_path + 'raw_log.bz2'):
        os.rename(data_path + 'raw_log.bz2', data_path + 'raw_log.bz2')

    df_sensors = load_sensor_data(data_path, offset=frame_offset)#.head(n_frames + 1)

    #roi_mat = load_transform_matrix(data_path + 'raw_log.bz2', start_time=df_sensors.loc[0, 't'])
    roi_mat = np.load(data_path + 'trns.npy')
    ext_mat = np.load(data_path + 'ext_mat.npy')

    if not os.path.exists(result_dir + 'result.json'):
        raise Exception('no file')

    with open(result_dir + 'result.json', 'r') as f:
        last_epoch = json.loads(f.read())['last_epoch']

    list_bgr_img = [cv2.imread(data_path + f'imgs/{i}.png') for i in range(frame_offset, frame_offset + n_frames + 1)]
    global_bev_mask = np.random.random((patch_length * scale, patch_width * scale)) > 0
    rb = ReplayBicycle(
        list_bgr_img, df_sensors, global_bev_mask, roi_mat, src_corners, scale=scale, target=model_type, ext_mat=ext_mat
    )
    cm = rb.run(start_steering_angle=None)
    #df_sensors.loc[:n_frames, 'lateral_shift_openpilot'] = [0] + cm.list_total_lateral_shift[:-1]
    #df_sensors.loc[:n_frames, 'yaw_openpilot'] = [0] + cm.list_yaw[:-1]

    #with open(result_dir + '/replay/model_benign_lane_pred.pkl', 'wb') as f:
    #    pickle.dump(rb.model_lane_pred, f, -1)

    cma_rep = CarMotionAttack(
        list_bgr_img,
        df_sensors,
        global_bev_mask,
        base_color,
        roi_mat,
        scale=scale,
        n_epoch=n_epoch,
        result_dir=result_dir,
        left_lane_pos=left_lane_pos,
        right_lane_pos=right_lane_pos,
        src_corners=src_corners,
        is_attack_to_rigth=is_attack_to_rigth,
        target_deviation=target_deviation,
        l2_weight=l2_weight,
        target=model_type,
        ext_mat=ext_mat,
    )

    cma_rep.calc_metric(
        epoch=last_epoch,
        target_frame=target_frame,
        starting_meters=starting_meters,
        lateral_shift=patch_lateral_shift,
        starting_steering_angle=cm.list_desired_steering_angle[0],
    )
    '''
    cma_rep.replay(
        epoch=last_epoch,
        starting_meters=starting_meters,
        lateral_shift=patch_lateral_shift,
        starting_steering_angle=cm.list_desired_steering_angle[0],
    )
    '''
def main(row, target_frame):

    config = {}    
    config['result_dir'] = f'logs/comma_attack/logs_more_{model_type}/{row.sc_name}/'
    config['data_path'] = f'comma2k19-ld/{row.sc_name}/'
    config['n_frames'] = 20
    config['frame_offset'] = 0
    config['target_frame'] = target_frame

    config['n_epoch'] = 200
    config['base_color'] = None#-0.1
    config['starting_meters'] = 7
    config['patch_lateral_shift'] = 1
    config['left_lane_pos'] = None
    config['right_lane_pos'] = None

    config['is_attack_to_rigth'] = int(row.sc_name[3:]) % 2 == 0
    config['patch_width'] = 30
    config['patch_length'] = 300
    config['l2_weight'] = 0.001

    #if os.path.exists(config['result_dir'] + os.path.basename(os.path.abspath(__file__)) + f'_f{target_frame}.log'):
    #    return

    os.makedirs(config['result_dir'] + '/replay/', exist_ok=True)

    for h in logger.handlers:
        logger.removeHandler(h)
    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    )

    logger.setLevel('INFO')


    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(
        config['result_dir'] + os.path.basename(os.path.abspath(__file__)) + f'_f{target_frame}.log', 'w'
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
        for target_frame in tqdm(range(20), ncols=55):
            main(row, target_frame)

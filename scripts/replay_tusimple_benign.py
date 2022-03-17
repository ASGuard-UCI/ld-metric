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
from car_motion_attack.load_sensor_data import load_sensor_data, load_transform_matrix, create_mock_driving

logger = getLogger(None)

model_type = sys.argv[1]

if model_type not in ('laneatt', 'ultrafast', 'scnn', 'polylanenet'):
    raise Exception(f'unknown model: {model_type}')

def read_img(path):
    img = np.zeros((874, 1164, 3), dtype=np.uint8)
    img[200:-220, 106:-106] = cv2.resize(cv2.imread(path), (952, 454))
    #img = cv2.resize(cv2.imread(path), (1164, 874))
    return img

def main(data_path='',
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
    if os.path.exists(result_dir + '/replay/model_benign_lane_pred.pkl'):
        return

    df_sensors = create_mock_driving(speed_ms=26.8224, n_frames=n_frames + 1) # 60 mph
    roi_mat = None
    list_bgr_img = [read_img(data_path + f'/{i + 1}.jpg') for i in range(frame_offset, frame_offset + n_frames + 1)]
    global_bev_mask = np.random.random((patch_length * scale, patch_width * scale)) > 0

    rb = ReplayBicycle(
        list_bgr_img, df_sensors, global_bev_mask, roi_mat, src_corners, scale=scale, target=model_type,
    )
    cm = rb.run(start_steering_angle=None, trajectory_update=False)
    #df_sensors['lateral_shift_openpilot'] = [0] + cm.list_total_lateral_shift[:-1]
    #df_sensors['yaw_openpilot'] = [0] + cm.list_yaw[:-1]

    with open(result_dir + '/replay/model_benign_lane_pred.pkl', 'wb') as f:
        pickle.dump(rb.model_lane_pred, f, -1)
    with open(result_dir + '/replay/model_benign_inputs.pkl', 'wb') as f:
        pickle.dump(rb.model_inputs, f, -1)
    with open(result_dir + '/replay/model_benign_outputs.pkl', 'wb') as f:
        pickle.dump(rb.model_outputs, f, -1)

if __name__ == '__main__':

    from logging import StreamHandler, Formatter, FileHandler
    config_path = sys.argv[2]

    with open(config_path, 'r') as f:
        config = json.loads(f.read())
        config['result_dir'] = f'logs/tusimple_benign/logs_{model_type}_replay/' + config['result_dir']
        config['l2_weight'] = 0.001
        config['n_epoch'] = 200
        
        #config['n_frames'] = 1
        #config['frame_offset'] = 15
        config['base_color'] = min(config['base_color'], -0.1)



    os.makedirs(config['result_dir'] + '/replay/', exist_ok=True)
    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    )

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(
        config['result_dir'] + os.path.basename(os.path.abspath(__file__)) + '.log', 'a'
    )
    handler.setLevel('DEBUG')
    handler.setFormatter(log_fmt)
    handler.setLevel('DEBUG')
    logger.addHandler(handler)

    logger.info(f'start: model={model_type}')
    main(**config)
    logger.info('end')


import numpy as np
import cv2
import torch
from scipy.interpolate import CubicSpline
from scnn.model import SCNN
from scnn.utils.transforms import Resize, Compose, Normalize, ToTensor

from car_motion_attack.config import PIXELS_PER_METER
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH,
                                      BEV_BASE_HEIGHT, BEV_BASE_WIDTH
                                      )

PIXELS_PER_METER_FOR_LANE = 7.928696412948382 + 1.2
MODEL_PATH_DISTANCE = 192
MODEL_OUTPUT_SIZE = 1760

def poly(x, coefs):
    return coefs[0] * x**3 + coefs[1] * x**2 + coefs[2] * x**1 + coefs[3]


def camera2model(_img):
    assert _img.shape == (874, 1164, 3)
    img = _img[200:-220, 106:-106]
    img = cv2.resize(img, (800, 288)).astype(np.float64)
    return img


def warp_coord(M, coord):
    if M.shape[0] == 3:
        x = (M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
        y = (M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
    else:
        x = M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2]
        y = M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2]

    warped_coord = np.array([x, y])
    return warped_coord


def get_line_points(img, th=0.3):
    assert img.shape == (288, 800)
    rows = []
    cols = []
    for i in range(18):
        row = int(288-(i)*20/590*288) - 1
        col = img[row, :].argmax()
        if img[row, col] > th:
            cols.append(col)
            rows.append(row)
    coords = np.array([cols,
                       rows]).T
    coords[:, 0] = coords[:, 0] / 800 * 952 + 106  # x
    coords[:, 1] = coords[:, 1] / 288 * 454 - 190  # y

    # pts = [warp_coord(car_motion.mtx_camera2bev,
    #                  (coord[0], coord[1])) for coord in coords]
    return coords


class OpenPilotSCNN:

    def __init__(self,
                 scale=5,
                 mtx_camera2bev=None,
                 weight_path = 'scnn/exp10_best.pth',
                 device='cpu'):

        self.scale = scale
        self.mtx_camera2bev = mtx_camera2bev
        self.device = devuce
        self.net = SCNN(input_size=(800, 288), pretrained=False)
        save_dict = torch.load(weight_path)
        self.net.load_state_dict(save_dict['net'])
        self.net.eval()
        self.net.to(self.device)

        self.transform_img = Resize((800, 288))
        self.transform_to_net = Compose(ToTensor(), Normalize(mean=(0.3598, 0.3653, 0.3662), 
                                                              std=(0.2573, 0.2663, 0.2756)))

    def predict(self, _img):
        #def scnn_predict_prefit(_img, car_motion, n_preds=192, scale=5, center_offset=-8):
        img = camera2model(_img)
        bev_shape = (BEV_BASE_HEIGHT * self.scale, BEV_BASE_WIDTH * self.scale)

        img = self.transform_img({'img': img})['img']
        x = self.transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        with torch.no_grad():
            self.net.to(self.device)
            x = x.to(self.device)
            #net.eval()
            seg_pred, exist_pred = self.net(x)[:2]
            seg_pred = seg_pred.cpu().numpy()[0]
            coord_mask = np.argmax(seg_pred, axis=0)
        #exist_pred = exist_pred.detach().cpu().numpy()
        #

        left_line = get_line_points(seg_pred[2])
        if left_line.shape[0] < 5:
            left_line = get_line_points(coord_mask == 2)
        right_line = get_line_points(seg_pred[3])
        if right_line.shape[0] < 5:
            right_line = get_line_points(coord_mask == 3)

        idx = left_line[:, 1].argsort()
        cs_left = CubicSpline(left_line[idx, 1], left_line[idx, 0])

        idx = right_line[:, 1].argsort()
        cs_right = CubicSpline(right_line[idx, 1], right_line[idx, 0])

        fixed_y = (bev_shape[0] - np.arange(1, MODEL_PATH_DISTANCE + 1) * PIXELS_PER_METER_FOR_LANE * self.scale).clip(0, bev_shape[0])

        pts = np.array([warp_coord(self.mtx_bev2camera,
                                (bev_shape[1] // 2, y)) for y in fixed_y])

        #pts = pts[pts[:, 1] > 5]
        left_line_pred = cs_left(pts[:, 1])
        right_line_pred = cs_right(pts[:, 1])

        left_line_bev = np.array([warp_coord(self.mtx_camera2bev,
                                            (left_line_pred[i], pts[i, 1])) for i in range(pts.shape[0])])
        right_line_bev = np.array([warp_coord(self.mtx_camera2bev,
                                            (right_line_pred[i], pts[i, 1])) for i in range(pts.shape[0])])

        center = bev_shape[1] // 2 + center_offset
        ppm = PIXELS_PER_METER_FOR_LANE * scale

        l_y = (center - left_line_bev[:, 0]) / ppm
        r_y = (center - right_line_bev[:, 0]) / ppm
        p_y = (l_y + r_y) / 2
        """
        n_pred = 40
        fixed_x = np.arange(0, n_preds)
        n_iter_x = n_preds - n_pred
        n_far = pts.shape[0] - n_pred

        l_y = np.array(l_y[:n_pred].tolist() + np.interp(np.arange(n_iter_x), np.linspace(0, n_iter_x, n_far), l_y[n_pred:]).tolist())
        r_y = np.array(r_y[:n_pred].tolist() + np.interp(np.arange(n_iter_x), np.linspace(0, n_iter_x, n_far), r_y[n_pred:]).tolist())
        p_y = np.array(p_y[:n_pred].tolist() + np.interp(np.arange(n_iter_x), np.linspace(0, n_iter_x, n_far), p_y[n_pred:]).tolist())
        """


        path_start = 0
        left_start = MODEL_PATH_DISTANCE * 2
        right_start = MODEL_PATH_DISTANCE * 2 + MODEL_PATH_DISTANCE * 2 + 1

        output = np.ones(MODEL_OUTPUT_SIZE)

        output[path_start:path_start + MODEL_PATH_DISTANCE] = p_y
        output[left_start:left_start + MODEL_PATH_DISTANCE] = l_y - 1.8
        output[right_start:right_start + MODEL_PATH_DISTANCE] = r_y + 1.8

        # return np.expand_dims(ouput, axis=0)
        return output

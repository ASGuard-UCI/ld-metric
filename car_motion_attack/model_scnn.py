import numpy as np
import cv2
import torch
from scipy.interpolate import CubicSpline
from scnn.utils.transforms import Resize, Compose, Normalize, ToTensor
from scnn.model import SCNN
from scipy.interpolate import InterpolatedUnivariateSpline

from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH,
                                      BEV_BASE_HEIGHT, BEV_BASE_WIDTH,
                                      INTRINSIC_MAT
                                      )
from car_motion_attack.utils import get_camera_points, get_bev_points
PIXELS_PER_METER_FOR_LANE = 7.928696412948382 + 1.2

N_PREDICTIONS = 192

IMG_W = 512
IMG_H = 288

from logging import getLogger
logger = getLogger(__name__)

def exp_line(pred):
    pred = torch.softmax(pred, axis=1)
    indices = (torch.arange(0, pred.shape[1]).float() / pred.shape[1]).to(pred.device)
    #indices = torch.stack([indices for _ in range(pred.shape[0])])

    argmax = (pred * indices).sum(axis=1)
    return argmax



def poly(x, coefs):
    return coefs[0] * x**3 + coefs[1] * x**2 + coefs[2] * x**1 + coefs[3]




def warp_coord(M, coord):
    if M.shape[0] == 3:
        x = (M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
        y = (M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
    else:
        x = M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2]
        y = M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2]

    warped_coord = np.array([x, y])
    return warped_coord


def get_line_points(img, th=0):
    assert img.shape == (IMG_H, IMG_W)
    rows = []
    cols = []
    for i in range(18):
        row = int(IMG_H - (i) * 20 / 590 * IMG_H) - 1
        col = img[row, :].argmax()
        if img[row, col] > th:
            cols.append(col)
            rows.append(row)
    coords = np.array([cols,
                       rows]).T
    coords[:, 0] = coords[:, 0] / IMG_W * 952 + 106  # x
    coords[:, 1] = coords[:, 1] / IMG_H * 454 + 200  # y

    # pts = [warp_coord(car_motion.mtx_camera2bev,
    #                  (coord[0], coord[1])) for coord in coords]
    return coords


class SCNNOpenPilot:

    def __init__(self,
                ext_mat,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                scale=5,
                center_offset=-8,
                weight_path=None,#'scnn/exp0_best.pth',
                is_attack_to_rigth=True,
                mtx_bev2camera=None
                ):
        self.ext_mat = ext_mat
        
        if mtx_bev2camera is not None:
            self.mtx_bev2camera = mtx_bev2camera
        else:
            self.mtx_bev2camera = np.dot(INTRINSIC_MAT, ext_mat)
        self.mtx_camera2bev = np.linalg.inv(self.mtx_bev2camera)


        self.device = device
        self.scale = scale
        self.center_offset = center_offset
        self.is_attack_to_rigth = is_attack_to_rigth

        self.net = SCNN(input_size=(IMG_W, IMG_H), pretrained=False)
        self.img_mean = (0.485, 0.456, 0.406)#(0.3598, 0.3653, 0.3662)  # CULane mean, std
        self.img_std = (0.229, 0.224, 0.225) #(0.2573, 0.2663, 0.2756)

        self.transform_img = Resize((IMG_W, IMG_H))
        self.transform_to_net = Compose(ToTensor(), Normalize(mean=self.img_mean, std=self.img_std))
        if weight_path is None:
            save_dict = torch.load('pretrained_models/scnn_exp0_best.pth', map_location=device)
        else:
            save_dict = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(save_dict['net'])
        self.net.eval()
        self.net.to(self.device)

        self.fixed_x = np.arange(N_PREDICTIONS) + 1 

    def camera2model(self, img):
        assert img.shape == (874, 1164, 3)
        img = img[200:-220, 106:-106]
        img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float64)
        return img

    def model2camera(self, model_img, camera_image=None):
        if camera_image is None:
            camera_image = np.zeros((874, 1164, 3))
        assert camera_image.shape == (874, 1164, 3)

        camera_image[200:-220, 106:-106] = cv2.resize(model_img, (952, 454))
        return camera_image

    def update_ext_mat(self, ext_mat):
        self.ext_mat = ext_mat
        
        self.mtx_bev2camera = np.dot(INTRINSIC_MAT, ext_mat)
        self.mtx_camera2bev = np.linalg.inv(self.mtx_bev2camera)
        
    def get_loss(self, img):
        #loss_func = softargmin
        
        img = self.camera2model(img)
        
        img = self.transform_img({'img': img})['img']
        x = self.transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        seg_pred, exist_pred = self.net(x)[:2]

        left_lane = seg_pred[0, 2]
        right_lane = seg_pred[0, 3]
        
        center = (exp_line(left_lane) + exp_line(right_lane)) / 2

        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center)
        else:
            loss = torch.mean(center)
        return loss.item()

    def get_avg_xs(self, img):
        #loss_func = softargmin
        
        img = self.camera2model(img)
        
        img = self.transform_img({'img': img})['img']
        x = self.transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        seg_pred, exist_pred = self.net(x)[:2]

        left_lane = seg_pred[0, 2]
        right_lane = seg_pred[0, 3]
        
        center = (exp_line(left_lane) + exp_line(right_lane)) / 2

        return center

    def get_input_gradient(self, img):
        #loss_func = softargmin
        
        img = self.camera2model(img)
        
        img = self.transform_img({'img': img})['img']
        x = self.transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        seg_pred, exist_pred = self.net(x)[:2]

        left_lane = seg_pred[0, 2]
        right_lane = seg_pred[0, 3]
        
        center = (exp_line(left_lane) + exp_line(right_lane)) / 2

        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center)
        else:
            loss = torch.mean(center)
        #print('AAA', loss.item())
        loss.backward()
        model_grad = x.grad[0].permute(1, 2, 0).cpu().numpy() * self.img_std * 255

        camera_grad = self.model2camera(model_grad)

        return camera_grad


    def predict(self, img):
        #np.save('orig_img', img)
        
        img = self.camera2model(img)
        
        img = self.transform_img({'img': img})['img']
        x = self.transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        x = x.to(self.device)
        with torch.no_grad():
            seg_pred, exist_pred = self.net(x)[:2]
            seg_pred = torch.sigmoid(seg_pred).cpu().numpy()[0]
            coord_mask = np.argmax(seg_pred[1:], axis=0)

        #logger.info(f'exist_pred: {exist_pred}')
        exist_pred = exist_pred.detach().cpu().numpy()

        self.seg_pred = seg_pred
        #np.save('seg_pred', seg_pred)

        # Get lane points in camera space
        left_line = get_line_points(seg_pred[2])
        #logger.info(f'left_line: {left_line.shape[0]}')
        if left_line.shape[0] < 10:
            left_line = get_line_points(coord_mask == (2 - 1))
        #logger.info(f'left_line: {left_line.shape[0]}')
        right_line = get_line_points(seg_pred[3])
        if right_line.shape[0] < 10:
            right_line = get_line_points(coord_mask == (3 - 1))

        self.left_line, self.right_line = left_line, right_line


        left_line_bev = get_bev_points(self.mtx_camera2bev, self.left_line)
        left_line_bev = left_line_bev[(left_line_bev[:, 0] > 0) & (left_line_bev[:, 0] < 50)]
        left_line_bev = left_line_bev[left_line_bev[:, 0].argsort()]
        right_line_bev = get_bev_points(self.mtx_camera2bev, self.right_line)
        right_line_bev = right_line_bev[right_line_bev[:, 0].argsort()]
        right_line_bev = right_line_bev[(right_line_bev[:, 0] > 0) & (right_line_bev[:, 0] < 50)]

        

        if 1:
            cs_left = InterpolatedUnivariateSpline(left_line_bev[:, 0], left_line_bev[:, 1], k=1, ext=3)
            cs_right = InterpolatedUnivariateSpline(right_line_bev[:, 0], right_line_bev[:, 1], k=1, ext=3)
            left_line_bev_pred = cs_left(self.fixed_x * 0.7)
            right_line_bev_pred = cs_right(self.fixed_x * 0.7)# + 0.15635729939444695
        else:
            cs_left = np.polyfit(left_line_bev[:, 0], left_line_bev[:, 1], deg=min(3, left_line_bev.shape[0]))
            cs_right = np.polyfit(right_line_bev[:, 0], right_line_bev[:, 1], deg=min(3, left_line_bev.shape[0]))
            left_line_bev_pred = np.polyval(cs_left, self.fixed_x * 0.7)
            right_line_bev_pred = np.polyval(cs_right, self.fixed_x * 0.7)# + 0.15635729939444695


        if left_line_bev_pred[0] < 0 or left_line_bev_pred[0] > 3:
            left_line_bev_pred = - right_line_bev_pred
        if right_line_bev_pred[0] > 0 or right_line_bev_pred[0] < -3:
            right_line_bev_pred = - left_line_bev_pred


        # Convert pixel to meter
        l_y = left_line_bev_pred#left_line_bev[:, 0] #(self.camera_center - left_line_bev[:, 0]) / self.ppm
        r_y = right_line_bev_pred#right_line_bev[:, 0] #(self.camera_center - right_line_bev[:, 0]) / self.ppm
        p_y = (l_y + r_y) / 2

        # Store in openpilot format
        path_start = 0
        left_start = N_PREDICTIONS * 2
        right_start = N_PREDICTIONS * 2 + N_PREDICTIONS * 2 + 1

        output = np.ones(1760)

        output[path_start:path_start + N_PREDICTIONS] = p_y
        output[left_start:left_start + N_PREDICTIONS] = l_y - 1.8
        output[right_start:right_start + N_PREDICTIONS] = r_y + 1.8

        # return np.expand_dims(ouput, axis=0)
        #np.save('output', output)
        return output

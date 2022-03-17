import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
import torch
import torchvision.transforms as transforms
from scipy.interpolate import InterpolatedUnivariateSpline

from functools import lru_cache
from lib.models import LaneATT
from lib.datasets import LaneDataset

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

from logging import getLogger
logger = getLogger(__name__)

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


def get_line_points(line):
    #line[:, 1] = 1 - line[:, 1]
    line = line * (952, 454) + [106, 200]
    return line


class LaneATTOpenPilot:

    def __init__(self,
                ext_mat,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                scale=5,
                center_offset=-8,
                img_size=(360, 640),
                is_attack_to_rigth=True,
                weight_path=None,
                mtx_bev2camera=None
                ):
        self.ext_mat = ext_mat
        
        if mtx_bev2camera is not None:
            self.mtx_bev2camera = mtx_bev2camera
        else:
            self.mtx_bev2camera = np.dot(INTRINSIC_MAT, ext_mat)


        self.device = device
        self.scale = scale
        self.center_offset = center_offset
        self.img_h, self.img_w = img_size
        self.is_attack_to_rigth = is_attack_to_rigth

        self.net = LaneATT(
                    backbone='resnet34',
                    S=72, 
                    topk_anchors=1000,
                    anchors_freq_path='LaneATT/data/tusimple_anchors_freq.pt',
                    img_h=self.img_h, 
                    img_w=self.img_w,
                    pretrained_backbone=True,
                    anchor_feat_channels=64,
                )
        if weight_path is None:
            self.net.load_state_dict(torch.load('pretrained_models/laneatt_r34_tusimplemodel_0100.pt', map_location=device)['model'])
        else:
            self.net.load_state_dict(torch.load(weight_path, map_location=device)['model'])

        self.net.to(self.device)
        self.net.eval()

        self.test_parameters = {'conf_threshold': 0.5, 'nms_thres': 45.0, 'nms_topk': 5}

        self.bev_shape = (BEV_BASE_HEIGHT * scale, BEV_BASE_WIDTH * scale)
        
        self.fixed_x = np.arange(N_PREDICTIONS) + 1 #(self.bev_shape[0] - np.arange(1, N_PREDICTIONS + 1) * PIXELS_PER_METER_FOR_LANE * scale).clip(0, self.bev_shape[0])

        #self.pts = get_trans_points(self.mtx_bev2camera, np.array([fixed_x, np.zeros_like(fixed_x)]).T)

        #self.camera_center = self.bev_shape[1] // 2 + self.center_offset

        #self.ppm = PIXELS_PER_METER_FOR_LANE * scale

        #self.list_left_points = 0

        self.left_line, self.right_line = None, None
        self.left_line_pred, self.right_line_pred = None, None
        #self.dataset = LaneDataset(
        #        S=72, 
        #        dataset='tusimple',
        #        split='test', 
        #        img_size=[360, 640], 
        #        max_lanes=5, 
        #        normalize=False, 
        #        aug_chance=0, 
        #        augmentations=None, 
        #        root='LaneATT/datasets/tusimple-test'
        #    )
    def update_ext_mat(self, ext_mat):
        self.ext_mat = ext_mat
        
        self.mtx_bev2camera = np.dot(INTRINSIC_MAT, ext_mat)
        self.mtx_camera2bev = np.linalg.inv(self.mtx_bev2camera)

    #@staticmethod
    def camera2model(self, img):
        assert img.shape == (874, 1164, 3)
        img = img[200:-220, 106:-106]
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 360)).astype(np.float32) / 255

        return img

    def model2camera(self, model_img, camera_image=None):
        if camera_image is None:
            camera_image = np.zeros((874, 1164, 3))
        assert camera_image.shape == (874, 1164, 3)

        camera_image[200:-220, 106:-106] = cv2.resize(model_img, (952, 454))
        return camera_image

    def get_loss(self, img):
        with torch.no_grad():
            x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
            x.unsqueeze_(0)

            x = x.to(self.device)
            #x.requires_grad = True

            cls_logits, reg = self.net.predict_for_grad(x)

            softmax = nn.Softmax(dim=1)
            #reg_proposals[0, :, :2] = softmax(reg_proposals[0, :, :2])
            lane_xs = reg[0, :, 1:] / 640#reg_proposals[0, :, 5:]#
            lane_conf = softmax(cls_logits[0, :, :])[:, 1] #reg_proposals[0, :, 1]
            #lane_start_y = reg_proposals[0, :, 2] * model.n_strips
            #lane_start_x = reg_proposals[0, :, 3]
            #lane_length = reg_proposals[0, :, 4]

            avg_xs = (lane_conf.reshape(-1, 1) * lane_xs).sum(axis=0) / lane_conf.sum()# expectation

            if self.is_attack_to_rigth:
                loss = torch.mean(1 - avg_xs)# * 100
            else:
                loss = torch.mean(avg_xs)# * 100

            return loss.item()

    def get_loss_multi(self, imgs):
        with torch.no_grad():
            x = torch.stack([torch.tensor(self.camera2model(img).transpose(2, 0, 1)) for img in imgs], axis=0)
            #x.unsqueeze_(0)

            x = x.to(self.device)
            #x.requires_grad = True

            cls_logits, reg = self.net.predict_for_grad(x)
            losses = []
            for i in range(len(imgs)):
                softmax = nn.Softmax(dim=1)
                #reg_proposals[0, :, :2] = softmax(reg_proposals[0, :, :2])
                lane_xs = reg[i, :, 1:] / 640#reg_proposals[0, :, 5:]#
                lane_conf = softmax(cls_logits[i, :, :])[:, 1] #reg_proposals[0, :, 1]
                #lane_start_y = reg_proposals[0, :, 2] * model.n_strips
                #lane_start_x = reg_proposals[0, :, 3]
                #lane_length = reg_proposals[0, :, 4]

                avg_xs = (lane_conf.reshape(-1, 1) * lane_xs).sum(axis=0) / lane_conf.sum()# expectation

                if self.is_attack_to_rigth:
                    loss = torch.mean(1 - avg_xs)# * 100
                else:
                    loss = torch.mean(avg_xs)# * 100
                losses.append(loss.item())
            del x
            return np.array(losses)

    def get_input_gradient(self, img):

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        cls_logits, reg = self.net.predict_for_grad(x)

        softmax = nn.Softmax(dim=1)
        #reg_proposals[0, :, :2] = softmax(reg_proposals[0, :, :2])
        lane_xs = reg[0, :, 1:] / 640#reg_proposals[0, :, 5:]#
        lane_conf = softmax(cls_logits[0, :, :])[:, 1] #reg_proposals[0, :, 1]
        #lane_start_y = reg_proposals[0, :, 2] * model.n_strips
        #lane_start_x = reg_proposals[0, :, 3]
        #lane_length = reg_proposals[0, :, 4]

        avg_xs = (lane_conf.reshape(-1, 1) * lane_xs).sum(axis=0) / lane_conf.sum()# expectation

        if self.is_attack_to_rigth:
            loss = torch.mean(1 - avg_xs) * 100
        else:
            loss = torch.mean(avg_xs) * 100

        loss.backward()
        #print('AAA', loss)
        model_grad = x.grad[0].permute(1, 2, 0).cpu().numpy()

        camera_grad = self.model2camera(model_grad)

        return camera_grad * 255


    def get_avg_xs(self, img):

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        cls_logits, reg = self.net.predict_for_grad(x)

        softmax = nn.Softmax(dim=1)
        #reg_proposals[0, :, :2] = softmax(reg_proposals[0, :, :2])
        lane_xs = reg[0, :, 1:] / 640#reg_proposals[0, :, 5:]#
        lane_conf = softmax(cls_logits[0, :, :])[:, 1] #reg_proposals[0, :, 1]
        #lane_start_y = reg_proposals[0, :, 2] * model.n_strips
        #lane_start_x = reg_proposals[0, :, 3]
        #lane_length = reg_proposals[0, :, 4]

        avg_xs = (lane_conf.reshape(-1, 1) * lane_xs).sum(axis=0) / lane_conf.sum()# expectation

        return avg_xs

    def draw_annotation(self, img, pred):
        for i, line in enumerate(pred):
            img = self.draw_line(img, line)
        return img

    def draw_line(self, img, line, color=(255, 0, 0)):
        try:
            points = line.points.copy()
        except AttributeError:
            points = line.copy()

        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        #points += pad
        xs, ys = points[:, 0], points[:, 1]
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                            tuple(curr_p),
                            tuple(next_p),
                            color=color,
                            thickness=3)
        return img


    def predict(self, img):
        #np.save('orig_img', img)

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x.unsqueeze_(0)

        x = x.to(self.device)
        with torch.no_grad():
            output = self.net(x, **self.test_parameters)
            prediction = self.net.decode(output, as_lanes=True)[0]
            if len(prediction) < 2:
                output = self.net(x, **{'conf_threshold': 0.01, 'nms_thres': 45.0, 'nms_topk': 5})
                prediction = self.net.decode(output, as_lanes=True)[0]
            #np.save('test', x.cpu().numpy())
            
            

            #import pdb;pdb.set_trace()
            #img = (x[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            #np.save('test', img)
            #img, fp, fn = self.dataset.draw_annotation(0, img=img, pred=prediction[0])
            #cv2.imshow('pred', img)
            #img = self.draw_annotation(img, prediction)
            #cv2.imwrite('text.jpg', img)
            #np.save('test2', img)

        
        def get_sloop(x):
            x = x.points[-1] - x.points[0]
            x = x[1] / x[0]
            return x

        #left_line = min(prediction, key=lambda x: 0.5 - x.points[:, 0].mean() if x.points[:, 0].mean() < 0.5 else np.inf).points#[::-1]
        #right_line = min(prediction, key=lambda x: x.points[:, 0].mean() - 0.5 if x.points[:, 0].mean() > 0.5 else np.inf).points#[::-1]

        left_line = min(prediction, key=lambda x: get_sloop(x)).points#[::-1]
        right_line = max(prediction, key=lambda x: get_sloop(x)).points#[::-1]

        left_line = get_line_points(left_line)
        right_line = get_line_points(right_line)

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

        #self.left_line_pred, self.right_line_pred = left_line_bev_pred, right_line_bev_pred

        #
        #np.save('left_line', left_line)
        #np.save('right_line', right_line)
        #np.save('all_lines', [get_line_points(p.points) for p in prediction])
        #np.save('left_line_bev', left_line_bev_pred)
        #np.save('right_line_bev', right_line_bev_pred)
        #import pdb;pdb.set_trace()

        # project to BEV


        #np.save('left_line', left_line_bev)
        #np.save('right_line', right_line_bev)
        #import pdb;pdb.set_trace()
        #left_line_bev = np.array([warp_coord(self.mtx_camera2bev,
        #                                    (left_line_pred[i], self.pts[i, 1])) for i in range(self.pts.shape[0])])
        #right_line_bev = np.array([warp_coord(self.mtx_camera2bev,
        #                                    (right_line_pred[i], self.pts[i, 1])) for i in range(self.pts.shape[0])])



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

        #np.save('output', output)
        
        #
        # return np.expand_dims(ouput, axis=0)
        return output

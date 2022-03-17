import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
import torch
import torchvision.transforms as transforms
from scipy.interpolate import InterpolatedUnivariateSpline

from PolyLaneNet.lib.models import PolyRegression

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


class PolyLaneNetOpenPilot:

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

        self.net = PolyRegression(
                num_outputs=35,
                pretrained=True, 
                backbone='efficientnet-b0',
                pred_category=False,
                curriculum_steps=[0, 0, 0, 0]
            )
        if weight_path is None:
            self.net.load_state_dict(torch.load('pretrained_models/polylanenet_model_2695.pt', map_location=device)['model'])
        else:
            self.net.load_state_dict(torch.load(weight_path, map_location=device)['model'])
        self.net.to(self.device)
        self.net.eval()

        self.img_transforms = transforms.Compose([
            #transforms.Resize((288, 800)),
            #transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.fixed_x = np.arange(N_PREDICTIONS) + 1 
        '''
        self.bev_shape = (BEV_BASE_HEIGHT * scale, BEV_BASE_WIDTH * scale)
        self.fixed_y = (self.bev_shape[0] - np.arange(1, N_PREDICTIONS + 1) * PIXELS_PER_METER_FOR_LANE * scale).clip(0, self.bev_shape[0])


        self.pts = np.array([warp_coord(self.mtx_bev2camera,
                                (self.bev_shape[1] // 2, y)) for y in self.fixed_y])


        self.camera_center = self.bev_shape[1] // 2 + self.center_offset

        self.ppm = PIXELS_PER_METER_FOR_LANE * scale

        self.list_left_points = 0
        '''    
    def update_ext_mat(self, ext_mat):
        self.ext_mat = ext_mat
        
        self.mtx_bev2camera = np.dot(INTRINSIC_MAT, ext_mat)
        self.mtx_camera2bev = np.linalg.inv(self.mtx_bev2camera)
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

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        _outputs = self.net(x)
        outputs, _ = self.net.decode(_outputs, None, conf_threshold=0.5)

        pred = outputs[0]
        pred = pred[pred[:, 0] != 0]
        poly = pred[:, 3:].mean(axis=0)

        xs = torch.arange(0.4, 1, 0.01).to(self.device)

        #loss = (3 * poly[0] * (xs ** 2) + 2 * poly[1] * xs + poly[2]).mean()
        center = (poly[0] * (xs ** 3) + poly[1] * (xs ** 2) + poly[2] * (xs) + poly[3])
        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center) 
        else:
            loss = torch.mean(center)

        return loss.item()

    def get_avg_xs(self, img):

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        _outputs = self.net(x)
        outputs, _ = self.net.decode(_outputs, None, conf_threshold=0.5)

        pred = outputs[0]
        pred = pred[pred[:, 0] != 0]
        poly = pred[:, 3:].mean(axis=0)

        xs = torch.arange(0.4, 1, 0.01).to(self.device)

        #loss = (3 * poly[0] * (xs ** 2) + 2 * poly[1] * xs + poly[2]).mean()
        center = (poly[0] * (xs ** 3) + poly[1] * (xs ** 2) + poly[2] * (xs) + poly[3])

        return center

    def get_input_gradient(self, img):

        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        _outputs = self.net(x)
        outputs, _ = self.net.decode(_outputs, None, conf_threshold=0.5)

        pred = outputs[0]
        pred = pred[pred[:, 0] != 0]
        poly = pred[:, 3:].mean(axis=0)

        xs = torch.arange(0.4, 1, 0.01).to(self.device)

        #loss = (3 * poly[0] * (xs ** 2) + 2 * poly[1] * xs + poly[2]).mean()
        center = (poly[0] * (xs ** 3) + poly[1] * (xs ** 2) + poly[2] * (xs) + poly[3])
        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center)# * 1000
        else:
            loss = torch.mean(center)# * 1000

        loss.backward()
        #print('AAA', loss.item())
        model_grad = x.grad[0].permute(1, 2, 0).cpu().numpy() * (0.229, 0.224, 0.225) * 255

        camera_grad = self.model2camera(model_grad)

        return camera_grad

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
        #np.save('inp_img', self.camera2model(img))
        x = torch.tensor(self.camera2model(img).transpose(2, 0, 1))
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x)
            outputs, _ = self.net.decode(outputs, None, conf_threshold=0.5)
            
            #import pdb;pdb.set_trace()
            #img = (x[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            #np.save('test', img)
            #img, fp, fn = self.dataset.draw_annotation(0, img=img, pred=prediction[0])
            #cv2.imshow('pred', img)
            #img = self.draw_annotation(img, prediction)
            #cv2.imwrite('text.jpg', img)
            #np.save('test2', img)

        pred = outputs[0].cpu().numpy()
        pred = pred[pred[:, 0] != 0]
        lower = pred[0, 1]
        predictions = []
        for i, lane in enumerate(pred):
            lane = lane[1:]  # remove conf
            _, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions
            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2))
            points[:, 1] = ys
            points[:, 0] = np.polyval(lane, ys)

            points = points[(points[:, 0] > 0) & (points[:, 0] < 1)]
            predictions.append(points)

        def get_sloop(x):
            x = x[-1] - x[0]
            x = x[1] / x[0]
            return x
        left_line = min(predictions, key=lambda x: get_sloop(x))#.points#[::-1]
        right_line = max(predictions, key=lambda x: get_sloop(x))#.points#[::-1]

        #left_line = min(predictions, key=lambda x: 0.5 - x[:, 0].mean() if x[:, 0].mean() < 0.5 else np.inf)
        #right_line = min(predictions, key=lambda x: x[:, 0].mean() - 0.5 if x[:, 0].mean() > 0.5 else np.inf)


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
        #import pdb;pdb.set_trace()
        # return np.expand_dims(ouput, axis=0)
        return output

import numpy as np
import cv2
import torch
import scipy
from scipy.interpolate import CubicSpline
from model.model import parsingNet
import torchvision.transforms as transforms

from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH,
                                      BEV_BASE_HEIGHT, BEV_BASE_WIDTH,
                                      INTRINSIC_MAT
                                      )
from scipy.interpolate import InterpolatedUnivariateSpline
from car_motion_attack.utils import get_camera_points, get_bev_points
PIXELS_PER_METER_FOR_LANE = 7.928696412948382 + 1.2

N_PREDICTIONS = 192

from logging import getLogger
logger = getLogger(__name__)


def softargmax(pred):
    pred = torch.softmax(pred, axis=1)
    indices = torch.range(0, pred.shape[1] - 1)
    indices = torch.stack([indices for _ in range(pred.shape[0])]).to(pred.device)

    argmax = (pred * indices).sum()

    return argmax

def softargmin(pred):
    return - softargmax(pred)


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
    assert img.shape == (288, 800)
    rows = []
    cols = []
    for i in range(18):
        row = int(288 - (i) * 20 / 590 * 288) - 1
        col = img[row, :].argmax()
        if img[row, col] > th:
            cols.append(col)
            rows.append(row)
    coords = np.array([cols,
                       rows]).T
    coords[:, 0] = coords[:, 0] / 800 * 952 + 106  # x
    coords[:, 1] = coords[:, 1] / 288 * 454 + 200  # y

    # pts = [warp_coord(car_motion.mtx_camera2bev,
    #                  (coord[0], coord[1])) for coord in coords]
    return coords


class UltraFastOpenPilot:

    def __init__(self,
                ext_mat,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                scale=5,
                center_offset=-8,
                weight_path=None,#'Ultra-Fast-Lane-Detection/tusimple_18.pth',
                is_attack_to_rigth=True,
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
        self.is_attack_to_rigth = is_attack_to_rigth


        self.img_transforms = transforms.Compose([
            #transforms.Resize((288, 800)),
            #transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        self.cls_num_per_lane = 56
        self.griding_num = 100
        self.row_anchor = np.array([64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108,
                                            112, 116, 120, 124, 128, 132, 136,
                                    140, 144, 148, 152, 156, 160, 164, 168, 172,
                                    176, 180, 184, 188, 192, 196, 200, 204, 
                                    208, 212, 216, 220, 224, 228, 232, 236, 240, 244,
                                    248, 252, 256, 260, 264, 268, 272, 276, 280, 284])

        self.col_sample = np.linspace(0, 800 - 1, self.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]

        self.net = parsingNet(pretrained = False,
                        backbone='18',
                        cls_dim = (self.griding_num+1, self.cls_num_per_lane, 4),
                            use_aux=False).to(self.device)
        if weight_path is None:
            state_dict = torch.load('pretrained_models/ultrafast_tusimple_18.pth', map_location='cpu')['model']
        else:
            state_dict = torch.load(weight_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()
        #self.net.to(self.device)

        self.fixed_x = np.arange(N_PREDICTIONS) + 1 

    def camera2model(self, img):
        assert img.shape == (874, 1164, 3)
        img = img[200:-220, 106:-106]
        img = cv2.resize(img, (800, 288)).astype(np.float64)
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
        
        img = self.camera2model(img)

        x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        out_j = self.net(x)[0]
        out_j = out_j.flip(1)

        prob = torch.nn.functional.softmax(out_j[:-1, :, :], dim=0) # row softmax
        idx = (torch.arange(self.griding_num).float() + 1) / self.griding_num
        idx = idx.reshape(-1, 1, 1).to(self.device)
        loc = torch.sum(prob * idx, axis=0) # expect xgrid
        
        
        left_lane = loc[:, 1]
        right_lane = loc[:, 2]

        center = (left_lane + right_lane) / 2

        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center)
        else:
            loss = torch.mean(center)
        return loss.item()

    def get_avg_xs(self, img):
        
        img = self.camera2model(img)

        x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        out_j = self.net(x)[0]
        out_j = out_j.flip(1)

        prob = torch.nn.functional.softmax(out_j[:-1, :, :], dim=0) # row softmax
        idx = (torch.arange(self.griding_num).float() + 1) / self.griding_num
        idx = idx.reshape(-1, 1, 1).to(self.device)
        loc = torch.sum(prob * idx, axis=0) # expect xgrid
        
        
        #left_lane = loc[:, 1]
        #right_lane = loc[:, 2]

        center = loc.mean(axis=0)#(left_lane + right_lane) / 2

        return center

    def get_input_gradient(self, img):
        
        img = self.camera2model(img)

        x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        x.requires_grad = True

        out_j = self.net(x)[0]
        out_j = out_j.flip(1)

        prob = torch.nn.functional.softmax(out_j[:-1, :, :], dim=0) # row softmax
        idx = (torch.arange(self.griding_num).float() + 1) / self.griding_num
        idx = idx.reshape(-1, 1, 1).to(self.device)
        loc = torch.sum(prob * idx, axis=0) # expect xgrid
        
        
        #left_lane = loc[:, 1]
        #right_lane = loc[:, 2]

        center = loc.mean(axis=0)#(left_lane + right_lane) / 2

        if self.is_attack_to_rigth:
            loss = torch.mean(1 - center)# * 1000
        else:
            loss = torch.mean(center)# * 1000

        loss.backward()

        model_grad = x.grad[0].permute(1, 2, 0).cpu().numpy() * (0.229, 0.224, 0.225) * 255

        camera_grad = self.model2camera(model_grad)

        return camera_grad


    def predict(self, img):
        #np.save('orig_img', img)
        img = self.camera2model(img)

        x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255
        x = self.img_transforms(x)
        x.unsqueeze_(0)

        x = x.to(self.device)
        
        with torch.no_grad():
            out = self.net(x)

        out_j = out[0].data.cpu().numpy()  # (101, 56, 4) = (xgrid, y, ch)
        out_j = out_j[:, ::-1, :] # y inverse
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # row softmax
        idx = np.arange(self.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0) # expect xgrid
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = np.nan
        out_j = loc

        def get_sloop(_x):
            x = _x[~np.isnan(_x)]
            if len(x) < 3:
                return 0
            x = x[-1] - x[0]
            x = (~np.isnan(_x)).sum() / x

            return x
        lane_loc = [get_sloop(out_j[:, i]) for i in range(out_j.shape[1])]
        #lane_loc = (out_j * self.col_sample_w  / 800).mean(axis=0)
        left = np.argmax(lane_loc) #np.argmin(np.where(lane_loc < 0.5, 0.5 - lane_loc, np.inf))
        right = np.argmin(lane_loc)#np.argmin(np.where(lane_loc > 0.5, lane_loc - 0.5, np.inf))

        left_line = np.vstack([
            out_j[:, left] * self.col_sample_w * 952 / 800 + 106,
            self.row_anchor[[self.cls_num_per_lane - 1 - k for k in range(out_j.shape[0])]] * 454 / 288 - 1 + 200
        ]).T
        left_line = left_line[out_j[:, left] > 0]
        #left_line[:, 1] = 288 - left_line[:, 1] 

        right_line = np.vstack([
            out_j[:, right] * self.col_sample_w * 952 / 800 + 106 ,
            self.row_anchor[[self.cls_num_per_lane - 1 - k for k in range(out_j.shape[0])]] * 454 / 288 - 1 + 200
        ]).T
        right_line = right_line[out_j[:, right] > 0]        
        
        self.left_line, self.right_line = left_line, right_line

        #if self.left_line.shape[0] * self.right_line.shape[0] == 0:
        #    import pdb;pdb.set_trace()


        left_line_bev = get_bev_points(self.mtx_camera2bev, self.left_line)
        left_line_bev = left_line_bev[(left_line_bev[:, 0] > 0) & (left_line_bev[:, 0] < 50)]
        left_line_bev = left_line_bev[left_line_bev[:, 0].argsort()]
        right_line_bev = get_bev_points(self.mtx_camera2bev, self.right_line)
        right_line_bev = right_line_bev[right_line_bev[:, 0].argsort()]
        right_line_bev = right_line_bev[(right_line_bev[:, 0] > 0) & (right_line_bev[:, 0] < 50)]

        if left_line_bev.shape[0] < 2:
            left_line_bev = right_line_bev * [1, -1]
        if right_line_bev.shape[0] < 2:
            right_line_bev = left_line_bev * [1, -1]

        cs_left = InterpolatedUnivariateSpline(left_line_bev[:, 0], left_line_bev[:, 1], k=1, ext=3)
        cs_right = InterpolatedUnivariateSpline(right_line_bev[:, 0], right_line_bev[:, 1], k=1, ext=3)

        left_line_bev_pred = cs_left(self.fixed_x * 0.7)
        right_line_bev_pred = cs_right(self.fixed_x * 0.7)# + 0.15635729939444695



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

        # return np.expand_dims(ouput, axis=0)
        return output

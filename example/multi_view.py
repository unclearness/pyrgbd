import cv2
import os
import json
import numpy as np
import sys
sys.path.append(os.path.abspath('.'))
import pyrgbd

def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_camera_params(kinect):
    param = {}
    param['depth'] = {}
    param['depth']['fx'] = kinect['K_depth'][0][0]
    param['depth']['fy'] = kinect['K_depth'][1][1]
    param['depth']['cx'] = kinect['K_depth'][0][2]
    param['depth']['cy'] = kinect['K_depth'][1][2]
    # ignore distCoeffs_depth's 5th (1000) and 6th (0) element
    # since it looks strange
    param['depth']['distCoeffs'] = np.array(kinect['distCoeffs_depth'][:5])

    param['color'] = {}
    param['color']['fx'] = kinect['K_color'][0][0]
    param['color']['fy'] = kinect['K_color'][1][1]
    param['color']['cx'] = kinect['K_color'][0][2]
    param['color']['cy'] = kinect['K_color'][1][2]
    # ignore distCoeffs_color's 5th (1000) and 6th (0) element
    # since it looks strange
    param['color']['distCoeffs'] = np.array(kinect['distCoeffs_color'][:5])

    d_T = np.array(kinect['M_depth'])
    c_T = np.array(kinect['M_color'])
    d2c_T = d_T @ c_T
    d2c_T = np.linalg.inv(d2c_T)
    
    param['d2c_R'] = d2c_T[0:3, 0:3]
    param['d2c_t'] = d2c_T[0:3, 3]

    w2d_T = np.array(kinect['M_world2sensor'])
    param['w2d_R'] = w2d_T[0:3, 0:3]
    param['w2d_t'] = w2d_T[0:3, 3]

    return param


if __name__ == '__main__':
    data_dir = './data/cmu_panoptic/171026_cello3/'
    kinect_params = load_json(os.path.join(data_dir,
                              'kcalibration_171026_cello3.json'))
    KINECT_NUM = 10
    for i in range(KINECT_NUM):
        param = parse_camera_params(kinect_params['sensors'][i])
        dfx, dfy, dcx, dcy = param['depth']['fx'], param['depth']['fy'], \
                             param['depth']['cx'], param['depth']['cy']
        cfx, cfy, ccx, ccy = param['color']['fx'], param['color']['fy'], \
                             param['color']['cx'], param['color']['cy']
        color = cv2.imread(os.path.join(
            data_dir, 'color_{:05d}.png'.format(i)))
        depth = cv2.imread(os.path.join(
            data_dir, 'depth_{:05d}.png'.format(i)), -1)
        depth = depth.astype(np.float) / 1000.0  # convert to meter scale
        mapped_color = pyrgbd.gen_mapped_color(depth, dfx, dfy, dcx, dcy,
                                               color, cfx, cfy, ccx, ccy,
                                               param['d2c_R'],  param['d2c_t'],
                                               ddist_type='OPENCV',
                                               ddist_param=
                                               param['depth']['distCoeffs'],
                                               cdist_type='OPENCV',
                                               cdist_param=
                                               param['color']['distCoeffs'])
        cv2.imwrite('mapped_{:05d}.png'.format(i), mapped_color)
        viz_depth = depth / 5.0
        viz_depth[viz_depth > 1.0] = 1.0
        viz_depth = (viz_depth * 255).astype(np.uint8)
        viz_depth = np.stack([viz_depth, viz_depth, viz_depth], axis=-1)
        print(viz_depth.shape, mapped_color.shape)
        mapped_color_with_depth = cv2.addWeighted(mapped_color, 0.3, viz_depth, 0.7, 0)
        cv2.imwrite('mapped_with_depth_{:05d}.png'.format(i), mapped_color_with_depth)
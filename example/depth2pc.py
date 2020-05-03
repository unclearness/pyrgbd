import cv2
import numpy as np 
import sys
import os
import time
sys.path.append(os.path.abspath('.'))
import pyrgbd

if __name__ == '__main__':
    data_dir = os.path.join('data', 'tum')
    color = cv2.imread(os.path.join(data_dir, 'color.png'), -1)
    color = color[:, :, [2, 1, 0]]  # BGR to RGB
    depth = cv2.imread(os.path.join(data_dir, 'depth.png'), -1)  # 16bit short
    depth = depth.astype(np.float32)
    depth /= 5000.0  # resolve TUM depth scale and convert to meter scale
    # intrinsics of Freiburg 3 RGB
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    start_t = time.time()
    pc, pc_color = pyrgbd.depth2pc(depth, fx, fy, cx, cy, color,
                                   keep_image_coord=False)
    end_t = time.time()
    print('depth2pc', end_t - start_t)

    '''
    # slow implementation
    start_t = time.time()
    pc, pc_color = pyrgbd.depth2pc_naive(depth, fx, fy, cx, cy, color)
    end_t = time.time()
    print('depth2pc_naive', end_t - start_t)
    '''

    start = time.time()
    pyrgbd.util.write_pc_ply_txt('pc.ply', pc, pc_color)
    end = time.time()
    print('write_pc_ply_txt', end - start)

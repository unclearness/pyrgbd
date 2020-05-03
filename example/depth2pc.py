import cv2
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath('.'))
import pyrgbd.util

if __name__ == '__main__':
    data_dir = os.path.join('data', 'tum')
    color = cv2.imread(os.path.join(data_dir, 'color.png'), -1)
    color = color[:, :, [2, 1, 0]]  # BGR to RGB
    depth = cv2.imread(os.path.join(data_dir, 'depth.png'), -1)  # 16bit short
    depth = depth.astype(np.float32)
    depth /= 5000.0  # resolve TUM depth scale and convert to meter scale
    # intrinsics of Freiburg 3 RGB
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    pc, pc_color = pyrgbd.util.depth2pc_naive(depth, fx, fy, cx, cy, color)
    pyrgbd.util.write_pc_ply_txt('pc.ply', pc, pc_color)

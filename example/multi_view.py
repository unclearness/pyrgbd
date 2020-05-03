import cv2
import os
import json
import numpy as np
import open3d as o3d
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
    # since they are strange
    param['depth']['distCoeffs'] = np.array(kinect['distCoeffs_depth'][:5])

    param['color'] = {}
    param['color']['fx'] = kinect['K_color'][0][0]
    param['color']['fy'] = kinect['K_color'][1][1]
    param['color']['cx'] = kinect['K_color'][0][2]
    param['color']['cy'] = kinect['K_color'][1][2]
    # ignore distCoeffs_color's 5th (1000) and 6th (0) element
    # since they are strange
    param['color']['distCoeffs'] = np.array(kinect['distCoeffs_color'][:5])

    d_T = np.array(kinect['M_depth'])
    c_T = np.array(kinect['M_color'])
    # "d_T @" is important...
    c2d_T = d_T @ c_T
    d2c_T = np.linalg.inv(c2d_T)

    param['d2c_R'] = d2c_T[0:3, 0:3]
    param['d2c_t'] = d2c_T[0:3, 3]

    # world to depth
    # "d_T @" is important...
    w2d_T = d_T @  np.array(kinect['M_world2sensor'])
    d2w_T = np.linalg.inv(w2d_T)
    param['w2d_R'] = w2d_T[0:3, 0:3]
    param['w2d_t'] = w2d_T[0:3, 3]
    param['w2d_T'] = w2d_T

    param['d2w_R'] = d2w_T[0:3, 0:3]
    param['d2w_t'] = d2w_T[0:3, 3]
    param['d2w_T'] = d2w_T

    return param


if __name__ == '__main__':
    data_dir = './data/cmu_panoptic/171026_cello3/'
    kinect_params = load_json(os.path.join(data_dir,
                                           'kcalibration_171026_cello3.json'))
    KINECT_NUM = 10
    global_pc = []
    global_pc_color = []

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.05,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    for i in range(KINECT_NUM):
        param = parse_camera_params(kinect_params['sensors'][i])
        dfx, dfy, dcx, dcy = param['depth']['fx'], param['depth']['fy'], \
            param['depth']['cx'], param['depth']['cy']
        cfx, cfy, ccx, ccy = param['color']['fx'], param['color']['fy'], \
            param['color']['cx'], param['color']['cy']
        ddist_param = param['depth']['distCoeffs']
        cdist_param = param['color']['distCoeffs']
        color = cv2.imread(os.path.join(
            data_dir, 'color_{:05d}.png'.format(i)))
        depth = cv2.imread(os.path.join(
            data_dir, 'depth_{:05d}.png'.format(i)), -1)
        depth = depth.astype(np.float) / 1000.0  # convert to meter scale
        mapped_color = pyrgbd.gen_mapped_color(depth, dfx, dfy, dcx, dcy,
                                               color, cfx, cfy, ccx, ccy,
                                               param['d2c_R'],  param['d2c_t'],
                                               ddist_type='OPENCV',
                                               ddist_param=ddist_param,
                                               cdist_type='OPENCV',
                                               cdist_param=cdist_param)
        # save mapped color
        cv2.imwrite('mapped_{:05d}.png'.format(i), mapped_color)
        viz_depth = depth / 5.0
        viz_depth[viz_depth > 1.0] = 1.0
        viz_depth = (viz_depth * 255).astype(np.uint8)
        viz_depth = np.stack([viz_depth, viz_depth, viz_depth], axis=-1)
        mapped_color_with_depth = \
            cv2.addWeighted(mapped_color, 0.3, viz_depth, 0.7, 0)
        cv2.imwrite('mapped_with_depth_{:05d}.png'.format(i),
                    mapped_color_with_depth)

        pc, pc_color = pyrgbd.depth2pc(
            depth, dfx, dfy, dcx, dcy, mapped_color, keep_image_coord=False)
        pc_color = pc_color[:, [2, 1, 0]]  # BGR to RGB

        # Merge Multiple Kinects into world_kinect coordinate (1st Kinect's coordinate)
        # TODO: Merge Multiple Kinects into panoptic_kinect coordinate
        pc = (param['d2w_R'] @ pc.T).T + param['d2w_t']

        pyrgbd.write_pc_ply_txt('pc_{:05d}.ply'.format(i), pc, pc_color)
        global_pc += pc.tolist()
        global_pc_color += pc_color.tolist()

        # BGR to RGB
        # copy() for C-style memory allocation after fancy indexing
        mapped_color_rgb = mapped_color[..., [2, 1, 0]].copy()
        o3d_color = o3d.geometry.Image(mapped_color_rgb)
        # to float32
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
        # TODO: Consider distortion
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_trunc=4.0, depth_scale=1.0,
            convert_rgb_to_intensity=False)
        h, w = depth.shape
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsic(w, h, dfx, dfy, dcx, dcy)),
            param['w2d_T'])

    global_pc = np.array(global_pc)
    global_pc_color = np.array(global_pc_color)
    pyrgbd.write_pc_ply_txt('pc_global.ply', global_pc, global_pc_color)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh_global_fusion.obj",
                               mesh,
                               write_triangle_uvs=True)

import numpy as np

import pyrgbd


def gen_mapped_color(depth, dfx, dfy, dcx, dcy,
                     color, cfx, cfy, ccx, ccy,
                     d2c_R, d2c_t,
                     ddist_type=None, ddist_param=[],
                     cdist_type=None, cdist_param=[],
                     cdist_interp='NN',
                     missing_color=[0, 0, 0]):
    # point cloud in depth camera coordinate
    dpc, _ = pyrgbd.depth2pc(depth, dfx, dfy, dcx, dcy,
                             keep_image_coord=True,
                             distortion_type=ddist_type,
                             distortion_param=ddist_param)
    # Valid region is depth > 0
    valid_mask = dpc[..., 2] > 0
    dpc = dpc[valid_mask]
    # point cloud in color camera coordinate
    cpc = (d2c_R @ dpc.T).T + d2c_t

    # Project to color camera coordinate
    img_p = pyrgbd.project(cpc[..., 0], cpc[..., 1], cpc[..., 2],
                           cfx, cfy, ccx, ccy, with_depth=False)
    u, v = img_p[..., 0], img_p[..., 1]

    with_cdist = cdist_type is not None
    if with_cdist:
        u, v = pyrgbd.undistort_pixel(u, v, cfx, cfy, ccx, ccy,
                                      cdist_type, cdist_param)

    # Interpolation for float uv by undistort_pixel
    if (with_cdist and cdist_interp == 'NN') or not with_cdist:
        v, u = np.rint(v).astype(np.int), np.rint(u).astype(np.int)
    elif with_cdist:
        raise NotImplementedError('cdist_interp ' +
                                  cdist_interp +
                                  ' is not implemented')
    dh, dw = depth.shape
    ch, cw, cc = color.shape

    # Guard if point cloud is projected to outside of color image
    # 1) Make UV valid mask for color
    v_valid = np.logical_and(0 <= v, v < ch)
    u_valid = np.logical_and(0 <= u, u < cw)
    uv_valid = np.logical_and(u_valid, v_valid)
    uv_invalid = np.logical_not(uv_valid)

    # 2) Set stub value for outside of valid mask
    v[v < 0] = 0
    v[(ch - 1) < v] = ch - 1
    u[u < 0] = 0
    u[(cw - 1) < u] = cw - 1

    # 3) Get color (with stub value)
    pc_color = color[v, u]

    # 4) Set missing_color for invalid region
    pc_color[uv_invalid] = missing_color

    # Prepare mapped color image
    mapped_color = np.zeros([dh, dw, cc], np.uint8)
    mapped_color[..., :] = missing_color

    # Update valid_mask
    # Set False for region where projection to color was failed
    all_false = np.zeros([dh, dw], np.bool)
    all_false[valid_mask] = uv_valid
    valid_mask = all_false

    # Set color to depth image coordinate
    # Note that this coordinate is BEFORE UNDISTORTION
    mapped_color[valid_mask] = pc_color[uv_valid]

    return mapped_color, valid_mask

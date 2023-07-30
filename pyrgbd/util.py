import numpy as np
import cv2


# https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
def _distort_pixel_opencv(u, v, fx, fy, cx, cy, k1, k2, p1, p2,
                          k3=0.0, k4=0.0, k5=0.0, k6=0.0):

    u1 = (u - cx) / fx
    v1 = (v - cy) / fy
    u2 = u1 ** 2
    v2 = v1 ** 2
    r2 = u2 + v2

    # https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L133
    _2uv = 2 * u1 * v1
    kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2)
    u_ = fx*(u1 * kr + p1 * _2uv + p2 * (r2+2*u2)) + cx
    v_ = fy*(v1 * kr + p1 * (r2 + 2*v2) + p2 * _2uv) + cy

    return u_, v_


def distort_pixel(u, v, fx, fy, cx, cy, distortion_type, distortion_param):
    if distortion_type == "OPENCV" and 4 <= len(distortion_param) <= 8:
        # k1, k2, p1, p2 = distortion_param
        return _distort_pixel_opencv(u, v, fx, fy,
                                     cx, cy, *tuple(distortion_param))
    raise NotImplementedError(
        distortion_type + " with param " + distortion_param
        + " is not implemented")


# https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
def _undistort_pixel_opencv(u, v, fx, fy, cx, cy, k1, k2, p1, p2,
                            k3=0.0, k4=0.0, k5=0.0, k6=0.0):
    # https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
    # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385
    x0 = (u - cx) / fx
    y0 = (v - cy) / fy
    x = x0
    y = y0
    # Compensate distortion iteratively
    # 5 is from OpenCV code.
    # I don't know theoritical rationale why 5 is enough...
    max_iter = 5
    for j in range(max_iter):
        r2 = x * x + y * y
        icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / \
            (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - deltaX) * icdist
        y = (y0 - deltaY) * icdist

    u_ = x * fx + cx
    v_ = y * fy + cy

    return u_, v_


def undistort_pixel(u, v, fx, fy, cx, cy, distortion_type, distortion_param):
    if distortion_type == "OPENCV" and 4 <= len(distortion_param) <= 8:
        # k1, k2, p1, p2 = distortion_param
        return _undistort_pixel_opencv(u, v, fx, fy,
                                       cx, cy, *tuple(distortion_param))
    raise NotImplementedError(
        distortion_type + " with param " + distortion_param
        + " is not implemented")


def undistort_depth(depth, fx, fy, cx, cy, distortion_type, distortion_param):
    # Undistortion for depth image
    # cv2.undistort uses bilinar interpolation
    # but it causes artifacts for boundary of depth image.
    # Nearest Neighbor is preferred.
    # This function provides NN like undistortion

    undistorted = np.zeros_like(depth)
    h, w = depth.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h), (w, 1)).T
    u, v = undistort_pixel(u, v, fx, fy, cx, cy,
                           distortion_type, distortion_param)
    v, u = np.rint(v).astype(int), np.rint(u).astype(int)

    # Make valid mask for original depth image space
    v_valid = np.logical_and(0 <= v, v < h)
    u_valid = np.logical_and(0 <= u, u < w)
    uv_valid = np.logical_and(u_valid, v_valid)
    uv_invalid = np.logical_not(uv_valid)

    # 0 for invalid
    depth[uv_invalid] = 0

    # Fiil stub
    v[v < 0] = 0
    v[(h - 1) < v] = h - 1
    u[u < 0] = 0
    u[(w - 1) < u] = w - 1

    # Copy depth value
    # Similar to Nearest Neighbor
    undistorted[v, u] = depth

    return undistorted


def medianBlurForDepthWithNoHoleFilling(depth, ksize):
    # cv2.medianBlur may fill invalid (0) for depth image
    # This function prevents it
    invalid_mask = depth == 0
    depth = cv2.medianBlur(depth, ksize)
    depth[invalid_mask] = 0
    return depth


def unproject(u, v, d, fx, fy, cx, cy):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    # for scalar and tensor
    return np.stack([x, y, d], axis=-1)


def project(x, y, z, fx, fy, cx, cy, with_depth=True):
    u = x * fx / z + cx
    v = y * fy / z + cy
    # for scalar and tensor
    if with_depth:
        return np.stack([u, v, z], axis=-1)
    return np.stack([u, v], axis=-1)


# TODO: faster version by tensor operation
def depth2pc_naive(depth, fx, fy, cx, cy, color=None, ignore_zero=True,
                   distortion_type=None, distortion_param=[],
                   distortion_interp='NN'):
    if depth.ndim != 2:
        raise ValueError()
    with_color = color is not None
    with_distortion = distortion_type is not None
    pc = []
    pc_color = []
    h, w = depth.shape
    for v in range(h):
        for u in range(w):
            d = depth[v, u]
            if ignore_zero and d <= 0:
                continue
            if with_distortion:
                u, v = undistort_pixel(u, v, distortion_type, distortion_param)
            p = unproject(u, v, d, fx, fy, cx, cy)
            pc.append(p)
            if with_color:
                # interpolation for float uv by undistort_pixel
                if with_distortion and distortion_interp == 'NN':
                    v, u = np.rint(v), np.rint(u)
                elif with_distortion:
                    raise NotImplementedError('distortion_interp ' +
                                              distortion_interp +
                                              ' is not implemented')
                if v < 0:
                    v = 0
                elif h - 1 < v:
                    v = h - 1
                if u < 0:
                    u = 0
                elif w - 1 < u:
                    u = w - 1
                pc_color.append(color[v, u])

    pc = np.array(pc)
    if with_color:
        pc_color = np.array(pc_color)
    return pc, pc_color


def depth2pc(depth, fx, fy, cx, cy, color=None, ignore_zero=True,
             keep_image_coord=False,
             distortion_type=None, distortion_param=[],
             distortion_interp='NN'):
    if depth.ndim != 2:
        raise ValueError()
    with_color = color is not None
    with_distortion = distortion_type is not None
    if ignore_zero:
        valid_mask = depth > 0
    else:
        valid_mask = np.ones(depth.shape, dtype=np.bool)
    invalid_mask = np.logical_not(valid_mask)
    h, w = depth.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h), (w, 1)).T
    if with_distortion:
        u, v = undistort_pixel(u, v, fx, fy, cx, cy,
                               distortion_type, distortion_param)
    pc = unproject(u, v, depth, fx, fy, cx, cy)
    pc[invalid_mask] = 0

    pc_color = None
    if with_color:
        # interpolation for float uv by undistort_pixel
        if with_distortion and distortion_interp == 'NN':
            v, u = np.rint(v), np.rint(u)
        elif with_distortion:
            raise NotImplementedError('distortion_interp ' +
                                      distortion_interp +
                                      ' is not implemented')

        # 1) Make UV valid mask for color
        v_valid = np.logical_and(0 <= v, v < h)
        u_valid = np.logical_and(0 <= u, u < w)
        uv_valid = np.logical_and(u_valid, v_valid)

        # 2) Set stub value for outside of valid mask
        v[v < 0] = 0
        v[(h - 1) < v] = h - 1
        u[u < 0] = 0
        u[(w - 1) < u] = w - 1
        pc_color = color[v, u]

        # 3) Update valid_mask and invalid_mask
        valid_mask = np.logical_and(valid_mask, uv_valid)
        invalid_mask = np.logical_not(valid_mask)

        pc_color[invalid_mask] = 0

    # return as organized point cloud keeping original image shape
    if keep_image_coord:
        return pc, pc_color

    # return as a set of points
    return pc[valid_mask], pc_color[valid_mask]


def _make_ply_txt(pc, color, normal):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(pc)),
                    "property float x", "property float y", "property float z"]
    has_normal = len(pc) == len(normal)
    has_color = len(pc) == len(color)
    if has_normal:
        header_lines += ["property float nx",
                         "property float ny", "property float nz"]
    if has_color:
        header_lines += ["property uchar red", "property uchar green",
                         "property uchar blue", "property uchar alpha"]
    # no face
    header_lines += ["element face 0",
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(pc)):
        line = [pc[i][0], pc[i][1], pc[i][2]]
        if has_normal:
            line += [normal[i][0], normal[i][1], normal[i][2]]
        if has_color:
            line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)
    data_txt = "\n".join(data_lines)

    # no face
    ply_txt = header + data_txt

    return ply_txt


def write_pc_ply_txt(path, pc, color=[], normal=[]):
    with open(path, 'w') as f:
        txt = _make_ply_txt(pc, color, normal)
        f.write(txt)


def visualize_depth(depth, mind, maxd, is_3channel=False):
    viz_depth = (depth - mind) / (maxd - mind)
    viz_depth[viz_depth > 1.0] = 1.0
    viz_depth[viz_depth < 0] = 0
    viz_depth = (viz_depth * 255).astype(np.uint8)
    if is_3channel:
        viz_depth = np.stack([viz_depth, viz_depth, viz_depth], axis=-1)
    return viz_depth

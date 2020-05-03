import numpy as np


# https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
def undistort_pixel_opencv(u, v, fx, fy, cx, cy, k1, k2, p1, p2,
                           k3=0.0, k4=0.0, k5=0.0, k6=0.0):
    u1 = u - cx
    v1 = v - cy
    u2 = u1 ** 2
    v2 = v1 ** 2
    r2 = u2 + v2
    r4 = r2 ** 2
    r6 = r2 ** 3
    radial_factor = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / \
        (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    u_ = u * radial_factor + 2 * p1 * u1 * v1 + p2 * (r2+2*u2)
    v_ = v * radial_factor + p1 * (r2 + 2*v2) + 2 * p2 * u1 * v1
    return u_, v_


def undistort_pixel(u, v, fx, fy, cx, cy, distortion_type, distortion_param):
    if distortion_type == "OPENCV" and len(distortion_param) == 4:
        k1, k2, p1, p2 = distortion_param
        return undistort_pixel_opencv(u, v, fx, fy, cx, cy, k1, k2, p1, p2)
    raise NotImplementedError(
        distortion_type + " with param " + distortion_param
        + " is not implemented")


def unproject(u, v, d, fx, fy, cx, cy, return_np=True):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    if return_np:
        return np.array([x, y, d])
    else:
        return [x, y, d]


def depth2pc_naive(depth, fx, fy, cx, cy, color=None, ignore_zero=True, return_np=True,
                   distortion_type=None, distortion_param=[]):
    if depth.ndim != 2:
        raise ValueError()
    with_color = color is not None
    pc = []
    pc_color = []
    h, w = depth.shape
    for v in range(h):
        for u in range(w):
            d = depth[v, u]
            if ignore_zero and d <= 0:
                continue
            if distortion_type is not None:
                u, v = undistort_pixel(u, v, distortion_type, distortion_param)
            p = unproject(u, v, d, fx, fy, cx, cy, return_np)
            pc.append(p)
            if with_color:
                # TODO: interpolation for float uv by undistort_pixel
                pc_color.append(color[v, u])
    if return_np:
        pc = np.array(pc)
        if with_color:
            pc_color = np.array(pc_color)
    return pc, pc_color


def make_ply_txt(pc, color, normal):
    header_lines = ["ply", "format ascii 1.0", "element vertex " + str(len(pc)),
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
        txt = make_ply_txt(pc, color, normal)
        f.write(txt)

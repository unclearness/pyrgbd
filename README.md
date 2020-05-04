# pyrgbd: python RGB-D processing tool
**pyrgbd** provides useful RGB-D processing snipets which are not directly covered by major libraries like OpenCV or Open3D.

## Implemented functions
- Project (3D -> 2D) / Unproject (2D with depth -> 3D)
- Depth image to point cloud
- Point cloud I/O
- Undistortion for depth image
  - `cv2.undistort()` causes artifacts for invalid (0) and boundary since it uses bilinear interpolation
  - Our implementation is artifact-free by using Nearest Neighbor like approach.
    - However, small lines or holes may occur

|Original|After undistortion|
|---|---|
|<img src=https://raw.githubusercontent.com/wiki/unclearness/pyrgbd/images/vis_depth_org_00000.png width=500>|<img src=https://raw.githubusercontent.com/wiki/unclearness/pyrgbd/images/vis_depth_undist_00000.png width=500>|

- RGB-D image coordinate mapping
  - By using intrinsics and extrinsics, convert RGB image onto depth image coordinate

|RGB|Depth|RGB mapped on Depth|
|---|---|---|
|<img src=https://raw.githubusercontent.com/wiki/unclearness/pyrgbd/images/color_00000.png width=500>|<img src=https://raw.githubusercontent.com/wiki/unclearness/pyrgbd/images/vis_depth_undist_00000.png width=500>|<img src=https://raw.githubusercontent.com/wiki/unclearness/pyrgbd/images/mapped_00000.png width=500>|



## Try
- Depth image to point cloud
  - `python example/depth2pc.py`

- Undistortion for depth image, RGB-D image coordinate mapping and multi-view integration
  - `python example/multi_view.py`
  - This example partially depends on Open3D 0.9.0

## Dependencies
- numpy
- cv2


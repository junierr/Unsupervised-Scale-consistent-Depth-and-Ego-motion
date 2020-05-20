from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MIN_DISP = 0.01

# TODO: delete test function
def show_shape(x):
    print(x.get_shape().as_list())

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  return transform_mat

# done!
def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width, _ = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  # [batch, 3, H*W]
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  # [B, 3, H, W]
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])
  # cam(x0, y0, z0) -> pixel(xp, yp)
  # target(xt, yt) -> cam(x0, y0, z0, h0) == cam_coords[batch, :, xt, yt]
  # -> pixel(xp, yp) == return[batch, xt, yt, :]

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.
  # 生成一个 2D 采样网络

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  # x_t [height, width] = [height, 1] * [1, width]
  # 每行的值为 [-1, 1] 区间，间隔数 width
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  # y_t [height, width] = [height, 1] * [1, width]
  # 每列的值为 [-1, 1] 区间，间隔数 height
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  # 映射坐标区间 [-1, 1] 到 [0, width - 1]
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  # 映射坐标区间 [-1, 1] 到 [0, height - 1]
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  # [batch, 2/3, height, width]
  return coords

def projective_inverse_warp(img, depth, pose, intrinsics):
  # 利用投影原理，将图片反投影到目标平面

  # img：源图像 [batch, height_s, width_s, 3]
  # depth：目标图像的深度图 [batch, height_t, width_t]
  # pose：目标来源相机变换矩阵[batch,6]，在 tx，ty，tz，rx，ry，rz的顺序
  # intrinsics：相机内参 [batch, 3, 3]
  # 返回： 源图像反向warp得到的目标图像平面 [batch, height_t, width_t, 3]
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()
  # Convert pose vector to matrix
  # 把相机运动向量转为 运动变换矩阵
  pose = pose_vec2mat(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # [batch, 3, height, width]
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # [batch, 4, height, width]
  # 从 像素坐标系 转换到 相机坐标系
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  # filler [batch, 1, 4]
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # intrinsics [batch, 4, 4]
  # [[*, *, *, 0],
  #  [*, *, *, 0],
  #  [*, *, *, 0],
  #  [0, 0, 0, 1]]
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  # 定义变换矩阵： 从target相机坐标系，转换到source像素坐标系
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  # cam(x0, y0, z0) -> pixel(xp, yp)
  # target(xs, ys) -> cam(x0, y0, z0, h0) == cam_coords[batch, :, xs, ys]
  # -> pixel(xt', yt') == return[batch, xs, ys, :]
  output_img = bilinear_sampler(img, src_pixel_coords)
  # 双线性内插法 得到 假 p_t
  return output_img

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output

def inverse_warp2(img, depth, ref_depth, pose, intrinsics):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    batch, height, width, _ = img.get_shape().as_list()
    # 将 Da转换到相机 a 镜头（归一坐标系）
    # Da 中 (x,y) 位置 对应于 相机坐标系的 坐标 (cam_coords[x, y])
    pixel_coords = meshgrid(batch, height, width)
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=False)
    # [B,H,W] -> [B,3,H,W]
    # a 到 b 相机运动
    pose_mat = pose_vec2mat(pose)  # [B,3,4]
    # b 的相机参数
    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)  # [B,3,4]
    # 旋转矩阵和平移矩阵
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    # 得到 Db 镜头的归一坐标系，得到 假 b (x,y)， 假 Db -> Db'
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr)  # [B,H,W,2]
    # I'a 假 Ib
    projected_img = bilinear_sampler(img, src_pixel_coords)
    # [batch, height, weight, channel]

    # 有效点
    #valid0 = tf.cast(src_pixel_coords[:, :, :, 0] <= width, dtype=tf.float32)
    #valid1 = tf.cast(src_pixel_coords[:, :, :, 0] >= 0, dtype=tf.float32)
    #valid2 = tf.cast(src_pixel_coords[:, :, :, 1] <= height, dtype=tf.float32)
    #valid3 = tf.cast(src_pixel_coords[:, :, :, 1] >= 0, dtype=tf.float32)
    #valid_points = valid0 * valid1 * valid2 * valid3
    valid0 = tf.cast(tf.abs(src_pixel_coords[:, :, :, 0]) <= (width), dtype=tf.float32)
    #print(valid0.get_shape().as_list())
    valid1 = tf.cast(tf.abs(src_pixel_coords[:, :, :, 1]) <= (height), dtype=tf.float32)
    #print(valid1.get_shape().as_list())
    #valid_points = tf.reduce_max(tf.abs(src_pixel_coords), axis=-1) <= 1
    valid_points = valid0 * valid1
    #print(valid_points.get_shape().as_list())
    #a = input()
    # valid_points = tf.reduce_max(tf.abs(src_pixel_coords), axis=-1) <= 1
    valid_mask = tf.cast(tf.expand_dims(valid_points, dim=3), dtype=tf.float32)
    projected_depth = bilinear_sampler(ref_depth, src_pixel_coords)
    # I'a, M, D'b, Dab
    return projected_img, valid_mask, projected_depth, computed_depth

def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.get_shape().as_list()
    cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = tf.matmul(proj_c2p_rot, cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
      pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2] + MIN_DISP

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    # X_norm = 2 * (X / Z) / (w - 1) - 1
    # Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    X_norm = (X / Z) 
    Y_norm = (Y / Z) 
    pixel_coords = tf.stack([X_norm, Y_norm], axis=2)  # [B, H*W, 2]
    return tf.reshape(pixel_coords, [b, h, w, 2]), tf.reshape(Z, [b, h, w, 1])
    # ref_img'  ref_depth'

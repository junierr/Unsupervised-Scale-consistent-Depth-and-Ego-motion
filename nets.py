from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

# TODO: delete test function
def show_shape(x):
    print(x.get_shape().as_list())

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    # 线性二次内插法
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def resize_fact(inputs, fac):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    if fac == 1:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [iH.value * fac, iW.value * fac])


def pose_net(tgt_image, src_image, is_training=True):
    with tf.variable_scope('pose_net', reuse=tf.AUTO_REUSE) as sc:
        H = tgt_image.get_shape()[1].value
        W = tgt_image.get_shape()[2].value
        imgs = tf.concat([tgt_image, src_image], axis=3)
        _, _, _, _, feature4 = res_encoder(imgs)
        # feature4
        # [batch, 4, 13, 512]
        # feature4
        # -------------------
        # DECODER
        squeeze = slim.conv2d(feature4, 256, 1, scope='squeeze')
        pose0 = slim.conv2d(squeeze, 256, 1, scope='pose0')
        pose1 = slim.conv2d(pose0, 256, 3, 1, scope='pose1')
        pose2 = slim.conv2d(pose1, 6, 1, activation_fn=None, scope='pose2')
        out = tf.reduce_mean(tf.reduce_mean(pose2, 2), 1)
        pose = 0.01 * tf.reshape(out, [-1, 6])
    return pose


def compute_pose_with_inv(tgt_img, ref_img_stack, is_training=True):
    poses = []
    poses_inv = []
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            for ref_img in ref_img_stack:
                poses.append(pose_net(tgt_img, ref_img, is_training=is_training))
                poses_inv.append(pose_net(ref_img, tgt_img, is_training=is_training))
    end_points = utils.convert_collection_to_dict(end_points_collection)
    return poses, poses_inv, end_points_collection


def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    # [batch, height, width, channel]
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    # 源图像数量
    num_source = int(src_image_stack.get_shape()[3].value // 3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            # [batch, 128, 416, channel]
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            # [batch, 64, 208, 16]
            cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
            # [batch, 32, 104, 32]
            cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
            # [batch, 16, 52, 64]
            cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            # [batch, 8, 26, 128]
            cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # [batch, 4, 13, 256]
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                # [batch, 2, 7, 256]
                cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                # [batch, 1, 4, 256]
                pose_pred = slim.conv2d(cnv7, 6 * num_source, [1, 1], scope='pred',
                                        stride=1, normalizer_fn=None, activation_fn=None)
                # 相机运动 6-DoF矩阵 [batch, 1, 4, 6 * num_source]
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # 相机运动 6-DoF矩阵 [batch, 6 * num_source]
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
                # pose [batch, num_source, 6]
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')
                    # [batch, 8, 26, 256]

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    # [batch, 16, 52, 128]
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4',
                                        normalizer_fn=None, activation_fn=None)
                    # [batch, 16, 52, num_source * 2]

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
                    # [batch, 32, 104, 64]
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3',
                                        normalizer_fn=None, activation_fn=None)
                    # [batch, 32, 104, num_source * 2]

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
                    # [batch, 64, 208, 32]
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2',
                                        normalizer_fn=None, activation_fn=None)
                    # [batch, 64, 208, num_source * 2]

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
                    # [batch, 128, 416, 16]
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1',
                                        normalizer_fn=None, activation_fn=None)
                    # [batch, 128, 416, num_source * 2]
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points
            # pose: [batch, num_source, 6]
            # mask1 [batch, 128, 416, num_source * 2]
            # mask2 [batch, 64, 208, num_source * 2]
            # mask3 [batch, 32, 104, num_source * 2]
            # mask4 [batch, 16, 52, num_source * 2]


# depth net
def disp_net(tgt_image, is_training=True):
    # tgt_img [batch, height, width, 3]
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # [batch, 128, 416, 3]
            cnv1 = slim.conv2d(tgt_image, 32, [7, 7], stride=2, scope='cnv1')
            # [batch, 64, 208, 32]
            cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
            # [batch, 64, 208, 32]
            cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
            # [batch, 32, 104, 64]
            cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
            # [batch, 32, 104, 64]
            cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            # [batch, 16, 52, 128]
            cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
            # [batch, 16, 52, 128]
            cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            # [batch, 8, 26, 256]
            cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
            # [batch, 8, 26, 256]
            cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            # [batch, 4, 13, 512]
            cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
            # [batch, 4, 13, 512]
            cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            # [batch, 2, 7, 512]
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            # [batch, 2, 7, 512]
            cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            # [batch, 1, 4, 512]
            cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')
            # [batch, 1, 4, 512]

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # [batch, 2, 8, 512]
            # There might be dimension mismatch due to uneven down/up-sampling
            # cnv6b:[batch, 2, 7, 512]
            upcnv7 = resize_like(upcnv7, cnv6b)
            # [batch, 2, 7, 512]
            i7_in = tf.concat([upcnv7, cnv6b], axis=3)
            # [batch, 2, 7, 1024]
            icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')
            # [batch, 2, 7, 512]

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            # [batch, 4, 14, 512]
            # cnv5b: [batch, 4, 13, 512]
            upcnv6 = resize_like(upcnv6, cnv5b)
            # [batch, 4, 13, 512]
            i6_in = tf.concat([upcnv6, cnv5b], axis=3)
            # [batch, 4, 13, 1024]
            icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            # [batch, 4, 13, 512]
            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            # [batch, 8, 26, 256]
            # cnv4b: [batch, 8, 26, 256]
            upcnv5 = resize_like(upcnv5, cnv4b)
            # [batch, 8, 26, 256]
            i5_in = tf.concat([upcnv5, cnv4b], axis=3)
            # [batch, 8, 26, 512]
            icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            # [batch, 8, 26, 256]
            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            # [batch, 16, 52, 128]
            # cnv3d: [batch, 16, 52, 128]
            i4_in = tf.concat([upcnv4, cnv3b], axis=3)
            # [batch, 16, 52, 256]
            icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            # [batch, 16, 52, 128]
            disp4 = DISP_SCALING * slim.conv2d(icnv4, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            # 输出 depth图像[batch, 16, 52, 1]
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])
            # resize [batch, 32, 104, 1]

            upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
            # [batch, 32, 104, 64]
            # conv2b: [batch, 32, 104, 64]
            i3_in = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            # [batch, 32, 104, 129]
            icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            # [batch, 32, 104, 64]
            disp3 = DISP_SCALING * slim.conv2d(icnv3, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            # 输出 depth图像[batch, 32, 104, 1]
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])
            # resize [batch, 64, 208, 1]

            upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
            # [batch, 64, 208, 32]
            # cnv1b: [batch, 64, 208, 32]
            i2_in = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            # [batch, 64, 208, 65]
            icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            # [batch, 64, 208, 32]
            disp2 = DISP_SCALING * slim.conv2d(icnv2, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            # 输出 depth图像[batch, 64, 208, 1]
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])
            # resize [batch, 128, 416, 1]

            upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
            # [batch, 128, 416, 16]
            # disp2_up: [batch, 128, 416, 1]
            i1_in = tf.concat([upcnv1, disp2_up], axis=3)
            # [batch, 128, 416, 17]
            icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
            # [batch, 128, 416, 16]
            disp1 = DISP_SCALING * slim.conv2d(icnv1, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            # 输出 depth图像[batch, 128, 416, 1]
            end_points = utils.convert_collection_to_dict(end_points_collection)
            # disp1: [batch, 128, 416, 1]
            # disp2: [batch, 64, 208, 1]
            # disp3: [batch, 32, 104, 1]
            # disp4: [batch, 16, 52, 1]
            return [disp1, disp2, disp3, disp4], end_points

def res_encoder(inputs):
    # inputs [batch, height, weight, 3 * n]
    feature0 = slim.conv2d(inputs, 64, [7, 7], stride=2, scope='convpre', normalizer_fn=slim.batch_norm)
    # feature0 [batch, 64, 208, 64]
    maxpool = slim.max_pool2d(feature0, kernel_size=3, stride=2, scope='maxpool', padding='SAME')
    # maxpool [batch, 32, 104, 64]

    blk1conv1 = slim.conv2d(maxpool, 64, [3, 3], stride=1, scope='blk1conv1',
                            normalizer_fn=slim.batch_norm)
    # blk1conv1 [batch, 32, 104, 64]
    blk1conv2 = slim.conv2d(blk1conv1, 64, [3, 3], stride=1, scope='blk1conv2',
                            normalizer_fn=slim.batch_norm)
    # blk1conv2 [batch, 32, 104, 64]
    blk1sample = maxpool
    # blk1sample [batch, 32, 104, 64]
    feature1 = tf.nn.relu(blk1sample + blk1conv2)
    # feature1 [batch, 32, 104, 64]

    blk2conv1 = slim.conv2d(feature1, 128, [3, 3], stride=2, scope='blk2conv1',
                            normalizer_fn=slim.batch_norm)
    # blk2conv1 [batch, 16, 52, 128]
    blk2conv2 = slim.conv2d(blk2conv1, 128, [3, 3], stride=1, scope='blk2conv2',
                            normalizer_fn=slim.batch_norm)
    # blk2conv2 [batch, 16, 52, 128]
    blk2sample = slim.conv2d(feature1, 128, [1, 1], stride=2, scope='blk2sample')
    # blk2sample [batch, 16, 52, 128]
    feature2 = tf.nn.relu(blk2sample + blk2conv2)
    # feature2 [batch, 16, 52, 128]

    blk3conv1 = slim.conv2d(feature2, 256, [3, 3], stride=2, scope='blk3conv1',
                            normalizer_fn=slim.batch_norm)
    # blk3conv1 [batch, 8, 26, 256]
    blk3conv2 = slim.conv2d(blk3conv1, 256, [3, 3], stride=1, scope='blk3conv2',
                            normalizer_fn=slim.batch_norm)
    # blk3conv2 [batch, 8, 26, 256]
    blk3sample = slim.conv2d(feature2, 256, [1, 1], stride=2, scope='blk3sample')
    # blk3sample [batch, 8, 26, 256]
    feature3 = tf.nn.relu(blk3sample + blk3conv2)
    # feature3 [batch, 8, 26, 256]

    blk4conv1 = slim.conv2d(feature3, 512, [3, 3], stride=2, scope='blk4conv1',
                            normalizer_fn=slim.batch_norm)
    # blk4conv1 [batch, 4, 13, 512]
    blk4conv2 = slim.conv2d(blk4conv1, 512, [3, 3], stride=1, scope='blk4conv2',
                            normalizer_fn=slim.batch_norm)
    # blk4conv2 [batch, 4, 13, 512]
    blk4sample = slim.conv2d(feature3, 512, [1, 1], stride=2, scope='blk4sample')
    # blk4sample [batch, 4, 13, 512]
    feature4 = tf.nn.relu(blk4sample + blk4conv2)
    # blk4sample [batch, 4, 13, 512]

    # features
    # [batch, 64, 208, 64]
    # [batch, 32, 104, 64]
    # [batch, 16, 52, 128]
    # [batch, 8, 26, 256]
    # [batch, 4, 13, 512]
    return feature0, feature1, feature2, feature3, feature4


def res_net(image, is_training=True, use_skips=True, name=''):
    # img [batch, height, width, 3]
    H = image.get_shape()[1].value
    W = image.get_shape()[2].value
    with tf.variable_scope('depth_net', reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = name + sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # ----------------------------
            # ENCODER
            # image [batch, 128, 416, 3*n]
            feature0, feature1, feature2, feature3, feature4 = res_encoder(image)
            # features
            # [batch, 64, 208, 64]
            # [batch, 32, 104, 64]
            # [batch, 16, 52, 128]
            # [batch, 8, 26, 256]
            # [batch, 4, 13, 512]
            # ----------------------------
            # ENCODER

            alpha = 10
            beta = 0.01
            # feature4 [batch, 4, 13, 512]
            upconv40 = ConvBlock(feature4, 256, scope='upconv40')
            # upconv40 [batch, 4, 13, 256]
            upsample4 = [resize_fact(upconv40, 2)]
            # upsample4 [batch, 8, 26, 256]
            if use_skips:
                upsample4 += [feature3]
                # feature3 [batch, 8, 26, 256]
            concat4 = tf.concat(upsample4, axis=3)
            upconv41 = ConvBlock(concat4, 256, scope='upconv41')
            # upconv41 [batch, 8, 26, 256]

            upconv30 = ConvBlock(upconv41, 128, scope='upconv30')
            # upconv30 [batch, 8, 26, 128]
            upsample3 = [resize_fact(upconv30, 2)]
            # upsample3 [batch, 16, 52, 128]
            if use_skips:
                upsample3 += [feature2]
                # feature2 [batch, 16, 52, 128]
            concat3 = tf.concat(upsample3, axis=3)
            upconv31 = ConvBlock(concat3, 128, scope='upconv31')
            # upconv31 [batch, 16, 52, 128]
            disp3 = alpha * tf.nn.sigmoid(ConvBlock(upconv31, 1, scope='dispconv3')) + beta
            # disp3 [batch, 16, 52, 1]

            upconv20 = ConvBlock(upconv31, 64, scope='upconv20')
            # upconv20 [batch, 16, 52, 64]
            upsample2 = [resize_fact(upconv20, 2)]
            # upsample2 [batch, 32, 104, 64]
            if use_skips:
                upsample2 += [feature1]
                # feature1 [batch, 32, 104, 64]
            concat2 = tf.concat(upsample2, axis=3)
            upconv21 = ConvBlock(concat2, 64, scope='upconv21')
            # upconv21 [batch, 32, 104, 64]
            disp2 = alpha * tf.nn.sigmoid(ConvBlock(upconv21, 1, scope='dispconv2')) + beta
            # disp2 [batch, 32, 104, 1]

            upconv10 = ConvBlock(upconv21, 32, scope='upconv10')
            # upconv10 [batch, 32, 104, 32]
            upsample1 = [resize_fact(upconv10, 2)]
            # upsample1 [batch, 64, 208, 32]
            if use_skips:
                upsample1 += [feature0]
                # feature0 [batch, 64, 208, 64]
            concat1 = tf.concat(upsample1, axis=3)
            upconv11 = ConvBlock(concat1, 32, scope='upconv11')
            # upconv11 [batch, 64, 208, 32]
            disp1 = alpha * tf.nn.sigmoid(ConvBlock(upconv11, 1, scope='dispconv1')) + beta
            # disp1 [batch, 64, 208, 1]

            upconv00 = ConvBlock(upconv11, 32, scope='upconv00')
            # upconv00 [batch, 64, 208, 32]
            upsample0 = resize_fact(upconv00, 2)
            # upsample0 [batch, 128, 416, 32]
            upconv01 = ConvBlock(upsample0, 32, scope='upconv01')
            # upconv01 [batch, 128, 416, 32]
            disp0 = alpha * tf.nn.sigmoid(ConvBlock(upconv01, 1, scope='dispconv0')) + beta
            # disp0 [batch, 128, 416, 1]

            end_points = utils.convert_collection_to_dict(end_points_collection)
            # disp0: [batch, 128, 416, 1]
            # disp1: [batch, 64, 208, 1]
            # disp2: [batch, 32, 104, 1]
            # disp3: [batch, 16, 52, 1]
            return [disp0, disp1, disp2, disp3], end_points


def ConvBlock(input, out_channels, use_refl=True, scope=None):
    if use_refl:
        out = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    else:
        out = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    out = slim.conv2d(out, out_channels, 3, padding='valid', activation_fn=tf.nn.elu, scope=scope)
    return out

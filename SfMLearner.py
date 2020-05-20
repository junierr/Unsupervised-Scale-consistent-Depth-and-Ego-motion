from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *

class SfMLearner(object):
    def __init__(self):
        pass

    def build_train_graph(self):
        loader_or_not = True
        opt = self.opt
        with tf.name_scope("data_loading"):
            if loader_or_not == True:
                loader = DataLoader(opt.dataset_dir,
                                    opt.batch_size,
                                    opt.img_height,
                                    opt.img_width,
                                    opt.num_source,
                                    opt.num_scales)
                tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            else:
                tgt_image, src_image_stack, intrinsics = get_fake_data(opt.batch_size,
                                                                       opt.img_height,
                                                                       opt.img_width,
                                                                       3,
                                                                       opt.num_source)
            tgt_image = preprecess_image(tgt_image)
            src_image_stack = preprecess_image(src_image_stack)
            ref_images = []
            num_source = int(src_image_stack.get_shape()[3].value // 3)
            for i in range(num_source):
                ref_images.append(src_image_stack[:, :, :, 3*i:3*(i+1)])
            # tgt_image [batch, height, weight, 3]
            # ref_images n * [batch, height, weight, 3]

        with tf.name_scope("depth_prediction"):
            pred_tgt_disp, pred_ref_disps = self.compute_depth(tgt_image, ref_images)
            # pred_tgt_disp:                        pred_ref_disps:
            # [disp1, disp2, disp3, disp4]          n * [disp1, disp2, disp3, disp4]
            # disp1: [batch, 128, 416, 1]
            # disp2: [batch, 64, 208, 1]
            # disp3: [batch, 32, 104, 1]
            # disp4: [batch, 16, 52, 1]

        with tf.name_scope("pose_and_pose_inv_prediction"):
            poses, poses_inv, _ = self.compute_pose_with_inv(tgt_image, ref_images, is_training=True)
            # poses:        n * [batch, 6]
            # poses_inv:    n * [batch, 6]

        with tf.name_scope("compute_loss"):
            loss1, loss3, mask, fake = self.compute_photo_and_geometry_loss(tgt_image, ref_images, intrinsics,
                            pred_tgt_disp, pred_ref_disps, poses, poses_inv)
            loss2 = self.compute_smooth_loss(tgt_image, pred_tgt_disp, ref_images, pred_ref_disps)
            # Loss1: L^M_P
            # Loss2: L_s
            # Loss3: L_GC
            loss =  opt.photo_weight * loss1 + \
                    opt.smooth_weight * loss2 + \
                    opt.geometry_weight * loss3

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optm = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1, opt.beta2)
            self.train_op = slim.learning.create_train_op(loss, optm)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)
            # step 自增函数，把 step+1 赋值给 step
        self.tgt_img = tgt_image
        self.ref_imgs = ref_images
        self.mask = mask
        self.fake = fake
        self.pred_tgt_disp = pred_tgt_disp
        self.pred_ref_disps = pred_ref_disps
        self.poses = poses
        self.poses_inv = poses_inv
        self.total_loss = loss
        self.photo_loss = loss1
        self.smooth_loss = loss2
        self.gemetry_loss = loss3
        if loader_or_not == True:
            self.steps_per_epoch = loader.steps_per_epoch
        else:
            self.steps_per_epoch = 10

    def compute_smooth_loss(self,
                            tgt_img,
                            tgt_depth,
                            ref_imgs,
                            ref_depths):
        def get_smooth_loss(disp, img):
            mean_disp = tf.reduce_mean(disp, axis=1, keep_dims=True)
            mean_disp = tf.reduce_mean(mean_disp, axis=2, keep_dims=True)
            norm_disp = disp / (mean_disp + 1e-7)
            disp = norm_disp

            grad_disp_x = tf.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
            grad_disp_y = tf.abs(disp[:, :-1, :, :] - disp[:, 1:, :, :])

            grad_img_x = tf.reduce_mean(tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), axis=3, keep_dims=True)
            grad_img_y = tf.reduce_mean(tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :]), axis=3, keep_dims=True)

            grad_disp_x *= tf.exp(-grad_img_x)
            grad_disp_y *= tf.exp(-grad_img_y)

            return tf.reduce_mean(grad_disp_x) + tf.reduce_mean(grad_disp_y)

        loss = get_smooth_loss(tgt_depth[0], tgt_img)

        for ref_depth, ref_img in zip(ref_depths, ref_imgs):
            loss += get_smooth_loss(ref_depth[0], ref_img)
        return loss

    def compute_photo_and_geometry_loss(self,
                                        tgt_img,
                                        ref_imgs,
                                        intrinsics,
                                        tgt_depth,
                                        ref_depths,
                                        poses,
                                        poses_inv):
        # tgt_img:图片 I_t
        # ref_img:图片 I_s
        # intrinsics:相机参数
        # tgt_depth:I_t
        # ref_depth:I_s
        # poses: I_t->I_s
        # poses_inv: I_s->I_t

        photo_loss = 0
        geometry_loss = 0
        num_scales = min(len(tgt_depth), self.opt.num_scales)
        # num_source
        for ref_img, ref_depth, pose, poses_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
            # num_scales
            for s in range(num_scales):
                tgt_img_scaled = tgt_img
                ref_img_scaled = ref_img
                intrinsic_scaled = intrinsics
                if s == 0:
                    tgt_depth_scaled = tgt_depth[s]
                    ref_depth_scaled = ref_depth[s]
                else:
                    tgt_depth_scaled = resize_like(tgt_depth[s], tgt_img)
                    ref_depth_scaled = resize_like(ref_depth[s], tgt_img)

                photo_loss1, geometry_loss1, mask, fake = self.compute_pairwise_loss(tgt_img_scaled, ref_img_scaled,
                                tgt_depth_scaled, ref_depth_scaled, pose, intrinsic_scaled)
                photo_loss2, geometry_loss2, mask, fake = self.compute_pairwise_loss(ref_img_scaled, tgt_img_scaled,
                                ref_depth_scaled, tgt_depth_scaled, poses_inv, intrinsic_scaled)
                photo_loss += photo_loss1 + photo_loss2
                geometry_loss += geometry_loss1 + geometry_loss2
        return photo_loss, geometry_loss, mask, fake

    def compute_pairwise_loss(self,
                              tgt_img,      # Ia
                              ref_img,      # Ib
                              tgt_depth,    # Da
                              ref_depth,    # Db
                              pose,         # Pab
                              intrinsic):
        ref_img_warped, valid_mask, projected_depth, computed_depth = \
            inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic)
        # ref_img_warped    I'a
        # valid_mask        V
        # projected_depth   D'b
        # computed_depth    Dab

        diff_img = tf.clip_by_value(tf.abs(tgt_img - ref_img_warped), clip_value_min=0., clip_value_max=1.)

        diff_depth = tf.clip_by_value(tf.abs(computed_depth - projected_depth) /
                                      tf.abs(computed_depth + projected_depth),
                                      clip_value_min=0., clip_value_max=1.)

        if self.opt.with_auto_mask == True:
            auto_mask = tf.less(tf.reduce_mean(diff_img, axis=3, keep_dims=True),
                                tf.reduce_mean(tf.abs(tgt_img - ref_img), axis=3, keep_dims=True)+0.05)
            auto_mask = tf.cast(auto_mask, dtype=tf.float32) * valid_mask
            valid_mask = auto_mask

        if self.opt.with_ssim == True:
            ssim_map = self.compute_ssim(tgt_img, ref_img_warped)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        if self.opt.with_mask == True:
            weight_mask = (1 - diff_depth)
            diff_img = diff_img * weight_mask

        photo_loss = self.mean_on_mask(diff_img, valid_mask)
        geometry_consistency_loss = self.mean_on_mask(diff_depth, valid_mask)

        return photo_loss, geometry_consistency_loss, valid_mask, ref_img_warped

    def mean_on_mask(self, diff, valid_mask):
        C = diff.get_shape()[3].value
        mask = tf.tile(valid_mask, [1, 1, 1, C])
        def fun1():
            return tf.reduce_sum(diff * mask) / tf.reduce_sum(mask)
        def fun2():
            return tf.cast(tf.zeros([]), dtype=tf.float32) + 1e-7
        mean_value = tf.cond(tf.reduce_sum(mask) > 10000,
                            fun1,
                            fun2)
        '''
        if tf.reduce_sum(mask) > 10000 is not None:
            mean_value = tf.reduce_sum(diff * mask) / tf.reduce_sum(mask)
        else:
            mean_value = tf.cast(tf.zeros([1]), dtype=tf.float32)
        '''

        return mean_value

    def compute_ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        with slim.arg_scope([slim.avg_pool2d],
                            kernel_size=[3, 3],
                            stride=1,
                            padding='SAME'):
            mu_x = slim.avg_pool2d(x)
            mu_y = slim.avg_pool2d(y)

            sigma_x = slim.avg_pool2d(x ** 2) - mu_x ** 2
            sigma_y = slim.avg_pool2d(y ** 2) - mu_y ** 2
            sigma_xy = slim.avg_pool2d(x * y) - mu_x * mu_y

            SSIM_n = (2 * mu_x *mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        return tf.clip_by_value((1 - SSIM_n / SSIM_d) / 2, clip_value_min=0., clip_value_max=1.)

    def compute_pose_with_inv(self, tgt_img, ref_imgs, is_training=True):
        # tgt_img [batch, height, width, 3]
        # reg_imgs n * [batch, height, width, 3]
        poses, poses_inv = [], []
        with tf.variable_scope('pose_and_pose_inv_net') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                for ref_img in ref_imgs:
                    poses.append(pose_net(tgt_img, ref_img, is_training=is_training))
                    poses_inv.append(pose_net(ref_img, tgt_img, is_training=is_training))
        end_points = utils.convert_collection_to_dict(end_points_collection)
        return poses, poses_inv, end_points_collection

    def compute_depth(self, tgt_img, ref_imgs):
        # tgt_img [batch, height, width, 3]
        # reg_imgs n * [batch, height, width, 3]
        tgt_disp, _ = res_net(tgt_img, is_training=True, name='tgt_disp_')
        pred_tgt_depth = [1./d for d in tgt_disp]

        pred_ref_depths = []
        for ref_img in ref_imgs:
            ref_disp, _ = res_net(ref_img, is_training=True, name='ref_disp_')
            pred_ref_depth = [1./d for d in ref_disp]
            pred_ref_depths.append(pred_ref_depth)
        return pred_tgt_depth, pred_ref_depths

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        # opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        print('-'* 30)
        a = input("check:")
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                    max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable Variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count = ", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,
                    "weight": tf.trainable_variables()[0]
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op
                    fetches["loss1"] = self.photo_loss
                    fetches["loss2"] = self.smooth_loss
                    fetches["loss3"] = self.gemetry_loss

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.6f loss1: %.6f loss2: %.6f loss3: %.6f" \
                          % (train_epoch, train_step, self.steps_per_epoch,
                             (time.time() - start_time) / opt.summary_freq,
                             results["loss"], results["loss1"], results["loss2"], results["loss3"]))
                    f = open(opt.checkpoint_dir + '/loss.txt', 'w')
                    f.write("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                          % (train_epoch, train_step, self.steps_per_epoch,
                             (time.time() - start_time) / opt.summary_freq,
                             results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    print(results["weight"][0][0][0])
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def collect_summaries(self):
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.photo_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.gemetry_loss)
        tf.summary.image('target_image', self.tgt_img)
        tf.summary.image("ref_image1", self.ref_imgs[0])
        tf.summary.image("mask", self.mask)
        tf.summary.image("fake", self.fake)
        for s in range(self.opt.num_scales):
            tf.summary.histogram("scale%d_depth" %s, self.pred_tgt_disp[s])
            tf.summary.image('scale%d_disparity_image' %s, 1./self.pred_tgt_disp[s])

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    # TODO: 待完成函数
    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width, 3], name='raw_input')
        input_mc = preprecess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            tgt_disp, _ = res_net(input_mc, is_training=True, name='tgt_disp_')
            pred_tgt_disp = [1. / d for d in tgt_disp]
        pred_depth = pred_tgt_disp[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth


    def build_pose_test_graph(self):
        pass

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        # if mode == 'pose':
        #     fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs: inputs})
        return results

def preprecess_image(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.
        # [-1, 1]
def deprocess_image(image):
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def get_fake_data(batch_size, height, width, channel, num_source):
    tgt_img = tf.ones([batch_size, height, width, channel], dtype=tf.float32) / 3.
    src_imgs = tf.ones([batch_size, height, width, channel*num_source], dtype=tf.float32) / 3.
    tgt_img = deprocess_image(tgt_img)
    src_imgs = deprocess_image(src_imgs)
    intrinsics = tf.eye(3, batch_shape=[batch_size])
    return tgt_img, src_imgs, intrinsics

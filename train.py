from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from SfMLearner import SfMLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/disks/disk2/guohao/qjh_use/dump_data", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "/disks/disk2/guohao/qjh_use/checkpoint2", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("beta2", 0.99, "beta of adam")
flags.DEFINE_float("photo_weight", 1.0, "Weight for photo loss")
flags.DEFINE_float("smooth_weight", 0.1, "Weight for smooth loss")
flags.DEFINE_float("geometry_weight", 0.5, "Weight for Geometry Consistency loss")
flags.DEFINE_boolean("with_ssim", True, "with SSIM or NOT")
flags.DEFINE_boolean("with_auto_mask", True, "with auto mask or NOT")
flags.DEFINE_boolean("with_mask", True, "with mask or NOT")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 832, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("num_source", 2, "Source of pictures for each example")
flags.DEFINE_integer("num_scales", 1, "Number of scales")
flags.DEFINE_integer("max_steps", 20001, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 1, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 40, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    sfm = SfMLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()

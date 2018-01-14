import os
import scipy.misc
import numpy as np

from model import Capsule
from utils import pp,  show_all_variables

import tensorflow as tf

#arguments parsers
flags = tf.app.flags

#name, defulat value, description
#below are for train and test
flags.DEFINE_integer("epoch", 5, "Epoch to train [25]")
flags.DEFINE_integer("test_epoch", 2000, "Epoch for latent mapping in anomaly detection to train [200]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("validation_check", False, "Use validation set and early stopping")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

flags.DEFINE_boolean("reconstruction_test", False, "In test, make reconstruction images [False]")
flags.DEFINE_boolean("tweak_test", False, "In test, make tweaked reconstruction images [False]")
flags.DEFINE_integer("tweak_num", 5, "Number of sample in tweak test [5]")

flags.DEFINE_integer("primary_dim", 8, "Dimensionality of Capsules in Primary Capsule Layer [8]")
flags.DEFINE_integer("digit_dim", 16, "Dimensionality of Capsules in Digit Capsule Layer [16]")
flags.DEFINE_integer("n_conv", 256, "Number of Channels in First Conv Layer [256]")
flags.DEFINE_integer("n_primary", 32, "Number of Capsules in Primary Capsule Layer [32]")
flags.DEFINE_integer("n_digit", 10, "Number of Capsules in Digit Capsule Layer. Same with number of classes [10]")


#below are for model construction
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value with input_height [None]")

flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, cifar10]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("val_checkpoint_dir", "val_checkpoint", "Directory name to save the checkpoints of early stop [val_checkpoint]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("test_dir", "test_data", "Directory name to load the anomaly detstion result [test_data]")

#flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if FLAGS.validation_check == True and not os.path.exists(FLAGS.val_checkpoint_dir):
    os.makedirs(FLAGS.val_checkpoint_dir)


  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  #run_config.gpu_options.per_process_gpu_memory_fraction = 0.4


  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      CapsuleNet = Capsule(sess, FLAGS)

    show_all_variables()

    if FLAGS.train:
      CapsuleNet.train()
    else:
      checkpoint_dir = FLAGS.val_checkpoint_dir if FLAGS.validation_check else FLAGS.checkpoint_dir
      if not CapsuleNet.load(checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")
      CapsuleNet.test_check()
      if FLAGS.reconstruction_test == True:
        CapsuleNet.test_reconstruction()
      if FLAGS.tweak_test == True:
        CapsuleNet.test_tweak(FLAGS.tweak_num)

if __name__ == '__main__':
  tf.app.run()

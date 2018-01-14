from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class Capsule(object):
  def __init__(self, sess, config, input_height=28, input_width=28, crop=True,
         batch_size=64,  c_dim=1, primary_dim=8, digit_dim=16, reg_para = 0.0005, n_conv=256,
         n_primary=32, n_digit=10, recon_h1=512, recon_h2=1024, dataset_name='mnist',
         input_fname_pattern='*.jpg', checkpoint_dir=None):

    self.config = config

    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size

    self.input_height = input_height
    self.input_width = input_width

    self.c_dim = c_dim

    self.primary_dim = primary_dim
    self.digit_dim = digit_dim

    self.n_conv = n_conv #256
    self.n_primary=n_primary
    self.n_digit = n_digit

    self.recon_h1 = recon_h1
    self.recon_h2 = recon_h2
    self.recon_output = self.input_height * self.input_width

    self.reg_para = reg_para

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.val_checkpoint_dir = config.val_checkpoint_dir

    self.load_mnist(self.config.validation_check)
    self.build_model()

  def build_model(self):
    self.primary_caps_layer = CapsConv(self.primary_dim, name='primary_caps')
    self.digit_caps_layer= CapsConv(self.digit_dim, name='digit_caps')


    self.input_x = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, self.c_dim], name='inputs')
    self.input_y = tf.placeholder(tf.float32, [None, self.n_digit], name='labels')
    self.recon_with_label = tf.placeholder_with_default(True, shape=(), name='reconstruction_with_label')

    self.conv1 = conv2d(self.input_x, self.n_conv, kernel_h=9, stride_h=1) #[batch_size, height, width, channel]
    self.primary_caps = self.primary_caps_layer(self.conv1, self.n_primary)
    self.digit_caps = self.digit_caps_layer(self.primary_caps, self.n_digit) ## shape: [batch_size, num_caps, dim_caps]

    with tf.variable_scope("prediction") as scope:
      self.logit = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=-1)) #[batch_size, num_caps]
      self.prob = tf.nn.softmax(self.logit)
      self.pred_label = tf.argmax(self.prob, axis=1)

    with tf.variable_scope("reconstruction") as scope:
      self.recon = self.reconstruction(name='reconstruction')

    with tf.variable_scope("loss") as scope:
      self.m_loss = margin_loss(self.logit, self.input_y)
      self.r_loss = reconstruction_loss(self.input_x, self.recon)
      self.loss = tf.add(self.m_loss, self.reg_para * self.r_loss)
      self.acc = self.accuracy(self.input_y, self.prob)

      self.m_loss_sum = tf.summary.scalar("margin_loss", self.m_loss)
      self.r_loss_sum = tf.summary.scalar("reconstruction_loss", self.r_loss)
      self.loss_sum = tf.summary.scalar("total_loss", self.loss)
      self.acc_sum = tf.summary.scalar("accuracy", self.acc)

    self.counter = tf.Variable(0, trainable=False)

    self.saver = tf.train.Saver()
    if self.validation_check == True:
      self.val_saver = tf.train.Saver()
      self.min_val_loss = np.inf

  def reconstruction(self, name='reconstruction'):
    if self.sess.run(self.recon_with_label) == True:
      mask_target = tf.argmax(self.input_y, axis=-1)
    else:
      mask_target = self.pred_label  #shape = [batch_size]

    recon_mask = tf.one_hot(mask_target, depth=self.n_digit, name='mask_output') #shape = [batch_size, 10]
    recon_mask = tf.reshape(recon_mask, [-1, self.n_digit, 1], name='reshape_mask_output') # shape [batch_size, 10 ,1]
    print recon_mask
    print self.digit_caps

    recon_mask = tf.multiply(self.digit_caps, recon_mask, name='mask_result')
    print recon_mask
    recon_mask = tf.layers.flatten(recon_mask, name='mask_input')

    with tf.variable_scope(name) as scope:
      hidden1 = fc_layer(recon_mask, self.recon_h1, activation='relu',name='hidden1')
      hidden2 = fc_layer(hidden1, self.recon_h2, activation='relu',name='hidden2')
      output = fc_layer(hidden2, self.recon_output, activation='sigmoid',name='reconstruction')

      return output
  
  def train(self):
    config = self.config
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    opt = optimizer.minimize(self.loss)
    tf.global_variables_initializer().run()

    self.summary_op = tf.summary.merge([self.m_loss_sum, self.r_loss_sum, self.loss_sum, self.acc_sum])

    batch_num = int(len(self.x_data)/self.batch_size)

    self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    for epoch in range(config.epoch):
      seed = 100
      np.random.seed(seed)
      np.random.shuffle(self.x_data)
      np.random.seed(seed)
      np.random.shuffle(self.y_data)
      for idx in range(batch_num):
        start_time = time.time()
        batch_x = self.x_data[idx*config.batch_size: (idx+1)*config.batch_size]
        batch_y = self.y_data[idx*config.batch_size: (idx+1)*config.batch_size]

        feed_dict = {self.input_x: batch_x, self.input_y: batch_y}

        _, loss, train_accuracy, summary_str =  self.sess.run([opt, self.loss, self.acc, self.summary_op], feed_dict=feed_dict) #add summary opt.
        total_count =  self.sess.run(self.counter.assign_add(1))
        self.writer.add_summary(summary_str, total_count)

        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, train_accurracy: %.8f" \
          % (epoch, idx, batch_num, time.time() - start_time, loss, train_accuracy))

        #check save
        if np.mod(total_count, batch_num) == batch_num-1:
          self.save(config.checkpoint_dir, total_count)
          if config.validation_check == True:
            self.validation_check(total_count)


  def validation_check(self, counter):
    assert self.config.validation_check == True
    feed_dict = {self.input_x: self.x_valid, self.input_y: self.y_valid}
    validation_loss, validation_accuracy= self.sess.run([self.loss, self.acc], feed_dict = feed_dict)
    print("Epoch: [%2d], validation_loss = %.8f, validation_accuracy: %.8f"\
      %(epoch, validation_loss, validation_accuracy))
    if validation_loss < self.min_val_loss:
      print("Checkpoint is saved for Early Stopping (min validation loss)")
      self.save(self.val_checkpoint_dir, counter)

  def test_check(self):
    test_num = int(len(self.x_test)/self.batch_size)
    test_loss, test_accuracy = 0.0, 0.0
    for idx in range(test_num-1):
      feed_dict = {self.input_x: self.x_test[idx*self.batch_size: (idx+1)*self.batch_size], self.input_y: self.y_test[idx*self.batch_size: (idx+1)*self.batch_size]}
      #feed_dict = {self.input_x: self.x_test, self.input_y: self.y_test}
      loss, accuracy = self.sess.run([self.loss, self.acc], feed_dict = feed_dict)
      test_loss+=loss
      test_accuracy+=accuracy
    test_loss /= test_num
    test_accuracy /= test_num
    print("[*] Final Results.. Test Loss: %.8f, Test Accuracy: %.8f" %(test_loss, test_accuracy))

  def test_reconstruction(self):
    num_recon = self.batch_size
    num_test = len(self.x_test)
    sample_idx = list(np.random.choice(num_test, num_recon))
    self.sample_x, self.sample_y = self.x_test[sample_idx], self.y_test[sample_idx]

    sample_frame_dim = int(math.ceil(self.config.batch_size**.5))
    if not os.path.isdir('./samples'): os.mkdir('./samples')
    save_images(self.sample_x, [sample_frame_dim, sample_frame_dim], './samples/samples_arrange.png')

    print "[*] Reconstruction images of samples are saved without labels"
    feed_dict = {self.input_x: self.sample_x, self.input_y: self.sample_y, self.recon_with_label: False}
    recon_images = self.sess.run(self.recon, feed_dict=feed_dict)

    recon_images = np.reshape(recon_images, [-1]+list(self.sample_x[0].shape))
    save_images(recon_images, [sample_frame_dim , sample_frame_dim], './samples/recon_samples_without_label_arrange.png')

    print "[*] Reconstruction images of samples are saved with labels"
    feed_dict = {self.input_x: self.sample_x, self.input_y: self.sample_y, self.recon_with_label: True}
    recon_images = self.sess.run(self.recon, feed_dict=feed_dict)
    recon_images = np.reshape(recon_images, [-1]+list(self.sample_x[0].shape))
    save_images(recon_images, [sample_frame_dim , sample_frame_dim], './samples/recon_samples_with_label_arrange.png')

  def load_mnist(self, valid=False):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY, dtype=np.int)
    teY = np.asarray(teY, dtype=np.int)

    #Make one-hot
    trY_vec = np.zeros((len(trY), self.n_digit))
    teY_vec = np.zeros((len(teY), self.n_digit))

    trY_index_offset = np.arange(len(trY)) * self.n_digit
    teY_index_offset = np.arange(len(teY)) * self.n_digit

    trY_vec.flat[trY_index_offset + trY.ravel()] = 1
    teY_vec.flat[teY_index_offset + teY.ravel()] = 1

    if not valid:
      self.x_data = trX/255.
      self.y_data = trY_vec
      self.x_test = teX/255.
      self.y_test = teY_vec
    else:
      seed=43
      np.random.seed(seed)
      np.random.shuffle(trX)
      np.random.seed(seed)
      np.random.shuffle(trY_vec)

      self.x_data = trX[:50000]/255.
      self.y_data = trY_vec[:50000]

      self.x_valid = trX[50000:]/255.
      self.y_valid = teY_vec[50000:]
      
      self.x_test = teX/255.
      self.y_test = teY_vec

  def accuracy(self, y, y_pred):
    #y: true one-hot label
    #y_pred: predicted logit
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    return accuracy

  @property
  def model_dir(self):
    return "{}_{}".format(
        self.dataset_name, self.batch_size,)
      
  def save(self, checkpoint_dir, step):
    model_name = "Capsule.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    if self.validation_check:
      print("[*] Reading optimal validation checkpoints...")

    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      #return True, counter
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt

from ops import *
from utils import *

class Capsule(object):
  def __init__(self, sess, config, input_height=28, input_width=None, crop=True,
         batch_size=64,  c_dim=1, primary_dim=8, digit_dim=16, reg_para = 0.0005, n_conv=256,
         n_primary=32, n_digit=10, recon_h1=512, recon_h2=1024, dataset_name='mnist',
         input_fname_pattern='*.jpg', checkpoint_dir=None):

    self.config = config

    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size

    self.input_height = input_height if not config.multi_MNIST else 36
    self.input_width = input_height if input_width == None else input_width

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
    if config.multi_MNIST == True:
      self.checkpoint_dir = config.multi_checkpoint_dir
    else:
      self.checkpoint_dir = config.checkpoint_dir
    
    self.val_checkpoint_dir = config.val_checkpoint_dir

    self.load_mnist(self.config.validation_check)
    self.build_model()

  def build_model(self):
    self.primary_caps_layer = CapsConv(self.primary_dim, name='primary_caps')
    self.digit_caps_layer= CapsConv(self.digit_dim, name='digit_caps')


    self.input_x = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, self.c_dim], name='inputs')
    self.input_y = tf.placeholder(tf.float32, [None, self.n_digit], name='labels')
    self.recon_with_label = tf.placeholder_with_default(True, shape=(), name='reconstruction_with_label')
    self.recon_multi = tf.placeholder_with_default(False, shape=(), name='reconstruction_with_multi_MNIST')

    self.conv1 = conv2d(self.input_x, self.n_conv, kernel_h=9, stride_h=1) #[batch_size, height, width, channel]
    self.primary_caps = self.primary_caps_layer(self.conv1, self.n_primary)
    self.digit_caps = self.digit_caps_layer(self.primary_caps, self.n_digit) ## shape: [batch_size, num_caps, dim_caps]

    with tf.variable_scope("prediction") as scope:
      self.logit = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=-1)) #[batch_size, num_caps]
      self.prob = tf.nn.softmax(self.logit)
      self.pred_label = tf.argmax(self.prob, axis=1)
      self.multi_pred_label = tf.nn.top_k(self.prob, 2).indices #for multiMNIST, [batch_size, 2]

    with tf.variable_scope("reconstruction") as scope:
      self.recon = self.reconstruction(multi_MNIST=self.config.multi_MNIST, name='reconstruction')

    with tf.variable_scope("loss") as scope:
      self.m_loss = margin_loss(self.logit, self.input_y)
      self.r_loss = reconstruction_loss(self.input_x, self.recon)
      self.loss = tf.add(self.m_loss, self.reg_para * self.r_loss)

      self.m_loss_sum = tf.summary.scalar("margin_loss", self.m_loss)
      self.r_loss_sum = tf.summary.scalar("reconstruction_loss", self.r_loss)
      self.loss_sum = tf.summary.scalar("total_loss", self.loss)
      if not self.config.multi_MNIST:
        self.acc = self.accuracy(self.input_y, self.prob)
        self.acc_sum = tf.summary.scalar("accuracy", self.acc)
      else:
        self.acc = None

    self.counter = tf.Variable(0, trainable=False)

    self.saver = tf.train.Saver()
    if self.validation_check == True:
      self.val_saver = tf.train.Saver()
      self.min_val_loss = np.inf

  def reconstruction(self, multi_MNIST = False, name='reconstruction'):
    if not self.sess.run(self.recon_multi):
      if self.sess.run(self.recon_with_label) == True:
        mask_target = tf.argmax(self.input_y, axis=-1)
      else:
        mask_target = self.pred_label  #shape = [batch_size]

      recon_mask = tf.one_hot(mask_target, depth=self.n_digit, name='mask_output') #shape = [batch_size, 10]
      recon_mask = tf.reshape(recon_mask, [-1, self.n_digit, 1], name='reshape_mask_output') # shape [batch_size, 10 ,1]

      recon_mask = tf.multiply(self.digit_caps, recon_mask, name='mask_result')
      recon_mask = tf.layers.flatten(recon_mask, name='mask_input')

    else:
      if self.sess.run(self.recon_with_label) == True:
        mask_target = tf.nn.top_k(self.input_y, 2).indices
      else:
        mask_target = self.multi_pred_label
    
      recon_mask_0 = tf.one_hot(mask_target[0], depth=self.n_digit, name='mask_output_0')
      recon_mask_0 = tf.multiply(self.digit_caps, recon_mask_0, name='mask_result_0')
      recon_mask_0 = tf.layers.flatten(recon_mask_0, name='mask_input_0')

      recon_mask_1 = tf.one_hot(mask_target[1], depth=self.n_digit, name='mask_output_1')
      recon_mask_1 = tf.multiply(self.digit_caps, recon_mask_1, name='mask_result_1')
      recon_mask_1 = tf.layers.flatten(recon_mask_1, name='mask_input_1')

      recon_mask = recon_mask_0 + recon_mask_1

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

    self.summary_op = tf.summary.merge_all()

    batch_num = int(len(self.x_data)/self.batch_size)
    #batch_num = 10

    self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
    
    could_load = self.load(self.checkpoint_dir)
    if could_load: print(" [*] Load SUCCESS")
    else: print (" [!] Load Failed...")

    for epoch in range(config.epoch):
      seed = 100
      np.random.seed(seed)
      np.random.shuffle(self.x_data)
      np.random.seed(seed)
      np.random.shuffle(self.y_data)

      if not self.config.multi_MNIST:

        for idx in range(batch_num-1):
          start_time = time.time()
          if not self.config.data_deformation:
            batch_x = self.x_data[idx*config.batch_size: (idx+1)*config.batch_size]
          else:
            batch_x = batch_deformation(self.x_data[idx*config.batch_size: (idx+1)*config.batch_size])

          batch_y = self.y_data[idx*config.batch_size: (idx+1)*config.batch_size]

          feed_dict = {self.input_x: batch_x, self.input_y: batch_y}

          _, loss, train_accuracy, summary_str =  self.sess.run([opt, self.loss, self.acc, self.summary_op], feed_dict=feed_dict) #add summary opt.
          total_count =  self.sess.run(self.counter.assign_add(1))
          self.writer.add_summary(summary_str, total_count)

          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, train_accurracy: %.8f" \
            % (epoch, idx, batch_num-1, time.time() - start_time, loss, train_accuracy))

          #check save
          if np.mod(total_count, batch_num) == batch_num-2:
            self.save(config.checkpoint_dir, total_count)
            if config.validation_check == True:
              self.validation_check(total_count)
      
      else:
        for idx in range(batch_num-2):
          start_time = time.time()
          batch_x1, batch_y1 = self.x_data[idx*config.batch_size: (idx+1)*config.batch_size], self.y_data[idx*config.batch_size: (idx+1)*config.batch_size]
          batch_x2, batch_y2 = self.x_data[(idx+1)*config.batch_size: (idx+2)*config.batch_size], self.y_data[(idx+1)*config.batch_size: (idx+2)*config.batch_size]
          batch_x, batch_y = multi_batch(batch_x1, batch_y1, batch_x2, batch_y2)

          feed_dict = {self.input_x: batch_x, self.input_y: batch_y}
          _, loss, summary_str = self.sess.run([opt, self.loss, self.summary_op], feed_dict=feed_dict)
          total_count = self.sess.run(self.counter.assign_add(1))
          self.writer.add_summary(summary_str, total_count)

          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f" \
            % (epoch, idx, batch_num-2, time.time() - start_time, loss))

          if np.mod(total_count, batch_num) == batch_num-3:
            self.save(config.multi_checkpoint_dir, total_count)

  def validation_check(self, counter):
    assert self.config.validation_check == True
    val_num = int(len(self.x_valid)/self.batch_size)
    val_loss, val_accuracy = 0.0, 0.0
    for idx in range(val_num-1):
      feed_dict = {self.input_x: self.x_valid[idx*self.batch_size: (idx+1)*self.batch_size], self.input_y: self.y_valid[idx*self.batch_size: (idx+1)*self.batch_size]}
      loss, accuracy = self.sess.run([self.loss, self.acc], feed_dict=feed_dict)
      val_loss += loss
      val_accuracy += accuracy
    
    val_loss /=(val_num-1)
    val_accuracy /= (val_num-1)
    print("[*] Validation: loss = %.8f, accuracy: %.8f"\
      %(validation_loss, validation_accuracy))

    if validation_loss < self.min_val_loss:
      print("[*] Checkpoint is saved for Early Stopping (min validation loss)")
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
    test_loss /= (test_num-1)
    test_accuracy /= (test_num-1)
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

  def test_tweak(self, tweak_sample=5, line_space=11):
    
    #tweak_range = np.linespace(-0.25, 0.25, line_space) #shape = [11]
    num_test = len(self.x_test)
    sample_idx = list(np.random.choice(num_test, tweak_sample))
    tweak_x, tweak_y = self.x_test[sample_idx], self.y_test[sample_idx]

    tweak_y = np.tile(tweak_y, [line_space*self.digit_dim,1]) # [5*11, 10], 
    

    feed_dict = {self.input_x: tweak_x} 
    sample_digit_caps = self.sess.run(self.digit_caps, feed_dict=feed_dict)

    steps = np.linspace(-0.25, 0.25, line_space)
    pose_paras = np.arange(self.digit_dim)

    tweaks = np.zeros([self.digit_dim, line_space, 1,1, self.digit_dim])
    tweaks[pose_paras, :, 0, 0, pose_paras] = steps #[16, 11, 1, 1, 16]
    sample_digit_caps = sample_digit_caps[np.newaxis, np.newaxis] #[16, 11, batch_size, 10, 16]

    tweaked_vectors = tweaks + sample_digit_caps #shape = [16, 11, batch_size, 10, 16]
    tweaked_vectors = np.reshape(tweaked_vectors, [-1, self.n_digit, self.digit_dim]) #[16*11*batch_size, 10, 16]

    tweaked_recon = self.sess.run(self.recon, feed_dict={self.digit_caps: tweaked_vectors, self.input_y: tweak_y, self.recon_with_label:False})
    tweaked_recon = np.reshape(tweaked_recon, [self.digit_dim, line_space, tweak_sample]+list(self.x_test[0].shape))

    if not os.path.isdir('./tweak_result'): os.mkdir('./tweak_result')
    for dim in range(self.digit_dim):
      plt.figure()
      for row in range(tweak_sample):
        for col in range(line_space):
          plt.subplot(tweak_sample, line_space, row * line_space + col + 1)
          cmap = 'binary' if np.shape(tweaked_recon)[-1] ==1 else None
          plt.imshow(np.squeeze(tweaked_recon[dim, col, row]), cmap=cmap)
          plt.axis('off')
      plt.savefig('./tweak_result/tweak_result_'+str(dim)+'.png')

  def test_multi_MNIST(self, multi_sample=4, num_iter=1):
    num_test = len(self.x_test)
    sample_idx1 = list(np.random.choice(num_test, multi_sample))
    sample_idx2 = list(np.random.choice(num_test, multi_sample))

    batch_x1, batch_y1 = self.x_test[sample_idx1], self.y_test[sample_idx1]
    batch_x2, batch_y2 = self.x_test[sample_idx2], self.y_test[sample_idx2]

    test_batch_x, test_batch_y = multi_batch(batch_x1, batch_y1, batch_x2, batch_y2)

    feed_dict = {self.input_x: test_batch_x, self.input_y: test_batch_y, self.recon_multi: True, self.recon_with_label: True}
    
    recon_images = self.sess.run(self.recon, feed_dict = feed_dict)
    recon_images = np.reshape(recon_images, [multi_sample] + list(np.shape(test_batch_x)[1:]))

    feed_dict = {self.input_x: test_batch_x, self.input_y: batch_y1}
    recon_images_1 = self.sess.run(self.recon, feed_dict = feed_dict)
    recon_images_1 = np.reshape(recon_images_1, [multi_sample] + list(np.shape(test_batch_x)[1:]))

    feed_dict = {self.input_x: test_batch_x, self.input_y: batch_y2}
    recon_images_2 = self.sess.run(self.recon, feed_dict = feed_dict)
    recon_images_2 = np.reshape(recon_images_2, [multi_sample] + list(np.shape(test_batch_x)[1:]))

    black = np.zeros(list(test_batch_x.shape))
    overlap_result = np.concatenate([black, recon_images_1, recon_images_2], axis=-1)
    image_1_result = np.concatenate([black, recon_images_1, black], axis=-1)
    image_2_result = np.concatenate([black, black, recon_images_2], axis=-1)


    #overlap_result = np.zeros(list(test_batch_x.shape)[:-1]+[3])
    #print np.shape(recon_images_1), np.shape(overlap_result[:,:,:,0]), np.shape(recon_images_1[:,:,:,:])
    #overlap_result[:,:,:,0] = recon_images_1
    #overlap_result[:,:,:,1] = recon_images_2

    if not os.path.isdir('./multi_MNIST_result'): os.mkdir('./multi_MNIST_result')
    for num in range(num_iter):
      plt.figure()
      cmap = 'binary'
      for idx in range(4):
        plt.subplot(4, multi_sample, idx+0*multi_sample+1)
        plt.imshow(test_batch_x[idx,:,:,0], cmap='gist_gray')
        #plt.imshow(test_batch_x[idx,:,:,0],cmap='gist_gray')
	plt.axis('off')

        plt.subplot(4, multi_sample, idx+1*multi_sample+1)
        plt.imshow(overlap_result[idx],cmap=cmap)
        #plt.imshow(overlap_result[idx,:,:,:])
        plt.axis('off')

        plt.subplot(4, multi_sample, idx+2*multi_sample+1)
        plt.imshow(image_1_result[idx], cmap=cmap)
        #plt.imshow(overlap_result[idx,:,:,0],cmap='gist_gray')
        plt.axis('off')

        plt.subplot(4, multi_sample, idx+3*multi_sample+1)
        plt.imshow(image_2_result[idx], cmap=cmap)
        #plt.imshow(overlap_result[idx,:,:,1], cmap='gist_gray')
        plt.axis('off')
      plt.savefig('./multi_MNIST_result/multi_test_'+str(num)+'.png')


  def load_mnist(self, valid=False):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
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
    if self.config.validation_check:
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
      return False

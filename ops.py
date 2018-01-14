import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)

#shape of vector: [None, ...(# of capsules)..., Caps-D]
def squash(vector,  axis=-1, epsilon=1e-5, name='squash_op'):
  with tf.name_scope(name) as scope:
    vec_norm_sq = tf.reduce_sum(tf.square(vector), axis=axis, keep_dims=True)
    vec_norm = tf.sqrt(vec_norm_sq+epsilon) #prevents from zero division
    scale = (vec_norm)/(1+vec_norm)
    squashed_vector = scale * (vector/vec_norm)
    return squashed_vector

def margin_loss(logit, pred_label, m_plus=0.9, m_minus=0.1, lamb=0.5):
  print logit
  print pred_label
  print tf.square(tf.maximum(0., m_plus-logit))

  loss = pred_label * tf.square(tf.maximum(0., m_plus-logit))
  loss += lamb* (1-pred_label) * tf.square(tf.maximum(0., logit-m_minus))
  return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

def reconstruction_loss(input_, recon):
  input_dim = input_.get_shape().as_list()[1:]
  len_dim = len(input_dim)
  input_size = 1
  for i in range(len_dim): input_size *=input_dim[i]

  input_ = tf.reshape(input_, [-1, input_size])
  recon = tf.reshape(recon, [-1, input_size])

  return tf.reduce_mean(tf.reduce_sum(tf.square(input_ - recon), axis=-1))

def conv2d(input_, output_dim, kernel_h=9, kernel_w=None, stride_h=2, stride_w=None, padding='VALID', reuse=False, initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv2d"):
  
  if kernel_w == None: kernel_w = kernel_h
  if stride_w == None: stride_w = stride_h

  with tf.variable_scope(name) as scope:
    if reuse==True: scope.reuse_variables()
    w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
      initializer=initializer)
    print w
    conv = tf.nn.conv2d(input_, w, strides=[1,stride_h, stride_w, 1], padding=padding)
    print conv
    b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, b)

    return conv

def fc_layer(input_, output_dim, initializer = tf.truncated_normal_initializer(stddev=0.02), activation='linear', reuse=False, name=None):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name or "Linear") as scope:
    if reuse==True: scope.reuse_variables()
    if len(shape) > 2 : input_ = tf.layers.flatten(input_)
    w = tf.get_variable("fc_w", [shape[1], output_dim], dtype=tf.float32, initializer = initializer)
    b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))

    result = tf.matmul(input_, w) + b

    if activation == 'linear':
      return result
    elif activation == 'relu':
      return tf.nn.relu(result)
    elif activation == 'sigmoid':
      return tf.nn.sigmoid(result)

class CapsConv(object):
  def __init__(self, d_caps, name=None,  is_train=True):
    self.d_caps = d_caps
    self.name = name if not name == None else "primary_caps"
    self.reuse = is_train

  def __call__(self, input_, n_caps, kernel_size=9, stride_size=2, route_iter=3, reuse=False, initializer = tf.truncated_normal_initializer(stddev=0.02)):
    input_shape = input_.get_shape().as_list()
    batch_size = tf.shape(input_)[0]
    #[None, height, width, channel]
    assert input_shape >= 2
    
    if 'primary' in self.name: #Primary Capsule
      with tf.variable_scope(self.name) as scope:
        if reuse==True: scope.reuse_variables()

        if len(input_shape) == 3: input_ = tf.expand_dims(input_,axis=-1) #for gray scale error
        self.input_shape = input_.get_shape().as_list()

        capsules = conv2d(input_, n_caps*self.d_caps, kernel_h=kernel_size, stride_h=stride_size, initializer = initializer)
        #capsule: [None, height, width, channel]
        capsule_shape = capsules.get_shape().as_list()
        total_num_cap = (capsule_shape[1]*capsule_shape[2]*capsule_shape[3])/self.d_caps
        # for preventing error from getting shape in digit caps. I defined the number of capsules to reshape

        capsules = tf.reshape(capsules, [batch_size,total_num_cap,self.d_caps])
        print capsules
        capsules = squash(capsules)
        print capsules
        return capsules

    else: #Digit Capsule
      with tf.variable_scope(self.name) as scope:
        if reuse==True: scope.reuse_variables()
        #let assume input_: [batch_size, # of capsules, d-capsules]
        #if len(input_shape) ==2 : input_ = tf.expand_dims(input_, axis=-1)
        self.input_shape = input_.get_shape().as_list()
        input_tiled = tf.expand_dims(input_, axis=-1, name='input_expand_1')#[batch_size, # of capsule, ..., d-capsules, 1]
        input_tiled = tf.expand_dims(input_tiled, axis=2, name='input_expand_2')#[batch_size, # of capsule, ..., d-capsules, 1]
        print input_tiled
        input_tiled = tf.tile(input_tiled, [1,1, n_caps, 1,1], name='input_tile')#[batch_size, # of capsule, # of next capsule, d_capsules, 1]
        print input_tiled

        W = tf.get_variable('prediction_w', [1,self.input_shape[1], n_caps, self.d_caps, self.input_shape[2]], initializer = initializer)
        W_tiled = tf.tile(W, [batch_size, 1,1,1,1], name = 'W_tiled')

        prediction_vectors = tf.matmul(W_tiled, input_tiled)

        b = tf.zeros([batch_size, self.input_shape[1], n_caps,1,1])

        for i in range(route_iter):
          coupling_coeff = tf.nn.softmax(b,dim=2)

          s = tf.multiply(prediction_vectors, coupling_coeff,name='weighted_prediction')
          sum_s = tf.reduce_sum(s, axis=1, keep_dims=True, name='weighted_sum')
          capsules = squash(sum_s, axis=-2) #(None, 1, # of nex capsule capsule, d_capsule, 1)
          caps_out_tile = tf.tile(capsules, [1, self.input_shape[1], 1,1,1], name='capsule_output_tiled')

          a = tf.matmul(prediction_vectors, caps_out_tile, transpose_a=True, name='agreement')
          b = tf.add(b, a, name='update_logit')

        capsules = tf.reshape(capsules, [batch_size, n_caps, self.d_caps])
        print capsules
        #return tf.squeeze(capsules)
        return capsules

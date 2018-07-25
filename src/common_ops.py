# -*- coding: utf-8 -*
#应该是包含了一些通用的操作
#
#
#create_weight()用来初始化权重
#create_bias()用来初始化偏置
import numpy as np
import tensorflow as tf


def lstm(x, prev_c, prev_h, w):
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)  #将x和prev_h拼合，并乘以w
  #ifog=[x,prev_h]*w,x和prev_h是以增广矩阵方式相连，即x的行数等于prev_h的行数
  #按列将ifog分成4份
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
	#使用keras接口，调用He正态分布初始化方法初始化权重
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
  if initializer is None:
	#偏置初始化为0.0
    initializer = tf.constant_initializer(0.0, dtype=tf.float32)
  return tf.get_variable(name, shape, initializer=initializer)


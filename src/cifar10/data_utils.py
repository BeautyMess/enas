# -*- coding: utf-8 -*
#用来读取CIFAR-10数据集中的images和labels，read_data()通过调用私有函数_read_data()实现读取数据的功能
import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print file_name
    full_name = os.path.join(data_path, file_name)
    with open(full_name) as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)           #所有图像样本聚合成一个长串的list
  labels = np.concatenate(labels, axis=0)           #所有标签聚合成一个长串的list
  images = np.reshape(images, [-1, 3, 32, 32])      #把图像reshape为[N,C,H,W]
  images = np.transpose(images, [0, 2, 3, 1])       #把[N,C,H,W]转换为[N,H,W,C]

  return images, labels


def read_data(data_path, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files) #使用私有函数

  #从训练集中抽取一部分作为验证集，数量为num_valids
  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  #读取测试集
  images["test"], labels["test"] = _read_data(data_path, test_file)

  
  print "Prepropcess: [subtract mean], [divide std]"
  #计算每个通道上的均值
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  #计算每个通道上的标准差
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print "mean: {}".format(np.reshape(mean * 255.0, [-1]))
  print "std: {}".format(np.reshape(std * 255.0, [-1]))

  """数据归一化"""
  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std
  """数据归一化"""

  return images, labels


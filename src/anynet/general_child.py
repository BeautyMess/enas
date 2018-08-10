#This file is responsible for generating child model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from src.anynet.models import Model
from src.anynet.image_ops import conv
from src.anynet.image_ops import fully_connected
from src.anynet.image_ops import batch_norm
from src.anynet.image_ops import batch_norm_with_mask
from src.anynet.image_ops import relu
from src.anynet.image_ops import max_pool
from src.anynet.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.common_ops import create_weight
from src.common_ops import create_bias



class GeneralChild(Model):
	def __init__(self,
				images,
				labels,
				cutout_size=None,
				whole_channels=False,
				fixed_arc=None,
				out_filters_scale=1,
				num_layers=2,
				num_branches=6,
				out_filters=24,
				keep_prob=1.0,
				batch_size=32,
				clip_mode=None,
				grad_bound=None,
				l2_reg=1e-4,
				lr_init=0.1,
				lr_dec_start=0,
				lr_dec_every=10000,
				lr_dec_rate=0.1,
				lr_cosine=False,
				lr_max=None,
				lr_min=None,
				lr_T_0=None,
				lr_T_mul=None,
				optim_algo=None,
				sync_replicas=False,
				num_aggregate=None,
				num_replicas=None,
				data_format="NHWC",
				name="child",
				*args,
				**kwargs
				):
		super(self.__class__, self).__init__(
				images,
				labels,
				cutout_size=cutout_size,
				batch_size=batch_size,
				clip_mode=clip_mode,
				grad_bound=grad_bound,
				l2_reg=l2_reg,
				lr_init=lr_init,
				lr_dec_start=lr_dec_start,
				lr_dec_every=lr_dec_every,
				lr_dec_rate=lr_dec_rate,
				keep_prob=keep_prob,
				optim_algo=optim_algo,
				sync_replicas=sync_replicas,
				num_aggregate=num_aggregate,
				num_replicas=num_replicas,
				data_format=data_format,
				name=name)

		self.whole_channels = whole_channels
		self.lr_cosine = lr_cosine
		self.lr_max = lr_max
		self.lr_min = lr_min
		self.lr_T_0 = lr_T_0
		self.lr_T_mul = lr_T_mul
		self.out_filters = out_filters * out_filters_scale
		self.num_layers = num_layers

		self.num_branches = num_branches
		self.fixed_arc = fixed_arc
		self.out_filters_scale = out_filters_scale

		pool_distance = self.num_layers // 3
		self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]
		#set the fixed size of hidden layer
		self.hidden_layer_size=500
		#set the output size of output layer and the number is the count of classes
		self.output_layer_size=10

	def _get_C(self, x):
		"""
		Args:
			x: tensor of shape [N, H, W, C] or [N, C, H, W]
		"""
		if self.data_format == "NHWC":
			return x.get_shape()[3].value
		elif self.data_format == "NCHW":
			return x.get_shape()[1].value
		else:
			raise ValueError("Unknown data_format '{0}'".format(self.data_format))

	def _get_HW(self, x):
		"""
		Args:
			x: tensor of shape [N, H, W, C] or [N, C, H, W]
		"""
		return x.get_shape()[2].value

	def _get_strides(self, stride):
		"""
		Args:
			x: tensor of shape [N, H, W, C] or [N, C, H, W]
		"""
		if self.data_format == "NHWC":
			return [1, stride, stride, 1]
		elif self.data_format == "NCHW":
			return [1, 1, stride, stride]
		else:
			raise ValueError("Unknown data_format '{0}'".format(self.data_format))

	"""Delete the content and leave a empty function"""

	def _factorized_reduction(self, x, out_filters, stride, is_training):
		final_path=[]
		return final_path


	def _get_C(self, x):
		"""
		Args:
			x: tensor of shape [N, H, W, C] or [N, C, H, W]
		"""
		if self.data_format == "NHWC":
			return x.get_shape()[3].value
		elif self.data_format == "NCHW":
			return x.get_shape()[1].value
		else:
			raise ValueError("Unknown data_format '{0}'".format(self.data_format))

	def _model(self, images, is_training, reuse=False):
		
		with tf.variable_scope(self.name, reuse=reuse):
			layers = []
			out_filters = self.out_filters
			"""
			reshape the input into [N,image vector size] for fully connected layer
			N=batch size
			and change the array "images" into tensor "x"
			"""
			w_or_h=np.shape(images)[2]
			x=tf.reshape(images,[-1,w_or_h*w_or_h])
			input_size=w_or_h*w_or_h
			hidden_size=self.hidden_layer_size
			with tf.variable_scope("input_layer"):
				w = create_weight("w", [input_size, hidden_size])
				b = create_bias("b",[1,hidden_size])
				x = tf.matmul(x, w)+b
				x = tf.nn.tanh(x)
				layers.append(x)
			print (layers[-1])
				
			#Just consider the default situation: whole_channels=True
			
			
			for layer_id in range(self.num_layers):
				with tf.variable_scope("hidden_layer_{0}".format(layer_id)):
					if self.fixed_arc is None:
						x = self._enas_layer(layer_id, layers, is_training)
					else:
						x = self._fixed_layer(layer_id, layers, is_training)
					layers.append(x)
					
					#have deleted the pool_at layer
					
				print(layers[-1])
				
			#in final code, it should be decide which layers will be connected to output_layer
			with tf.variable_scope("output_layer"):
				w = create_weight("w", [hidden_size, self.output_layer_size])
				#x = tf.matmul(x, w)
				b = create_bias("b",[1,self.output_layer_size])
				x = tf.matmul(x, w)+b
				x = tf.nn.tanh(x)
			print (x)
		
		return x

	def _enas_layer(self, layer_id, prev_layers, is_training):
		"""
		Args:
			layer_id: current layer
			prev_layers: cache of previous layers. for skip connections
			start_idx: where to start looking at. technically, we can infer this
				from layer_id, but why bother...
			is_training: for batch_norm
		"""
		#decide what the input is
		"""
		curr_idx: the position of this layer in prev_layers[]
		because prev_layers[0] resprensents the input layer, curr_idx should be added 1
		"""
		curr_idx=layer_id+1
		with tf.variable_scope("concat_input"):
			input_list=[]
			#meet the return requirement by using the function append_with_return()
			def append_with_return(inp_list,temp_layer):
				inp_list.append(temp_layer)
				return 1
			
			def return_0():
				return prev_layers[0]
			
			with tf.variable_scope("cond_1"):
				for temp_layer_id in range(layer_id):
				
					tf.cond(tf.equal(self.sample_arc[temp_layer_id], layer_id),
							lambda: append_with_return(input_list,prev_layers[temp_layer_id+1]),   #because prev_layers[0] resprensents the input layer, temp_layer_id should be added 1
							lambda: 0           #do nothing
							)
			with tf.variable_scope("cond_2"):
				print ("out1",prev_layers[0])
				print ("out2",np.shape(input_list))
				print ("out3",np.shape(tf.reduce_mean(input_list,0)))
				tf.cond(tf.equal(len(input_list),0),
							lambda: append_with_return(input_list,prev_layers[0]),
							lambda: 0
							)
				inputs=tf.reduce_mean(input_list,0)
				
				print ("out4",np.shape(inputs))
				print ("out5",inputs)
				print ("out6",type(inputs))
				print ("out7",np.shape(prev_layers[0]))
			""""
			for temp_layer_id in range(layer_id):
				
				tf.cond(tf.equal(self.sample_arc[temp_layer_id], layer_id),
						lambda: append_with_return(input_list,prev_layers[temp_layer_id+1]),   #because prev_layers[0] resprensents the input layer, temp_layer_id should be added 1
						lambda: 0           #do nothing
						)
			
			
			inputs=tf.cond(tf.equal(len(input_list),0),
							lambda: prev_layers[0],
							lambda: tf.reduce_mean(input_list,0)
							)
			inputs.set_shape([None, 500])
			"""
		with tf.variable_scope("FC"):
			w = create_weight("w", [self.hidden_layer_size, self.hidden_layer_size])
			b = create_bias("b",[1,self.hidden_layer_size])
			out = tf.matmul(inputs, w)+b
			out = tf.nn.tanh(out)
		return out

	def _fixed_layer(self, layer_id, prev_layers, is_training):
		"""
		Args:
			layer_id: current layer
			prev_layers: cache of previous layers. for skip connections
			start_idx: where to start looking at. technically, we can infer this
				from layer_id, but why bother...
			is_training: for batch_norm
		"""
		arc_seq=self.sample_arc
		"""
		input_list stroes layers that point to the current layer
		"""
		with tf.variable_scope("concat_input"):
			input_list=[]
			for temp_layer_id in range(layer_id):
				if arc_seq[temp_layer_id]== layer_id:
					input_list.append(prev_layers[temp_layer_id+1])   #because the input_layer has been added to prev_layers, the index of prev_layers should be added 1
			if input_list:
				#inputs = input_list[-1]         #in final code, it should be replaced with average or other method to use all inputs
				inputs=tf.reduce_mean(input_list,0)        #average all the input
			else:
				inputs=prev_layers[0]
			
		"""
		if self.whole_channels:
		#just consider the default setting: whole_channels=true
		"""
		with tf.variable_scope("FC"):
			w = create_weight("w", [self.hidden_layer_size, self.hidden_layer_size])
			b = create_bias("b",[1,self.hidden_layer_size])
			out = tf.matmul(inputs, w)+b
			out = tf.nn.tanh(out)
			#out = tf.matmul(inputs, w)
		"""
		count = self.sample_arc[start_idx]
		if count ==0:
			with tf.variable_scope("FC"):
				w = create_weight("w", [self.hidden_layer_size, self.hidden_layer_size])
				b = create_bias("b",[1,self.hidden_layer_size])
				out = tf.matmul(inputs, w)+b
				out = tf.nn.tanh(out)
				#out = tf.matmul(inputs, w)
		else:
			raise ValueError("Unknown operation number '{0}'".format(count))
		"""
		return out

	# override
	def _build_train(self):
		print("-" * 80)
		print("Build train graph")
		logits = self._model(self.x_train, is_training=True)
		log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_train)
		self.loss = tf.reduce_mean(log_probs)

		self.train_preds = tf.argmax(logits, axis=1)
		self.train_preds = tf.to_int32(self.train_preds)
		self.train_acc = tf.equal(self.train_preds, self.y_train)
		self.train_acc = tf.to_int32(self.train_acc)
		self.train_acc = tf.reduce_sum(self.train_acc)

		tf_variables = [var
			for var in tf.trainable_variables() if var.name.startswith(self.name)]
		self.num_vars = count_model_params(tf_variables)
		print("Model has {} params".format(self.num_vars))

		self.global_step = tf.Variable(
			0, dtype=tf.int32, trainable=False, name="global_step")
		self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
			self.loss,
			tf_variables,
			self.global_step,
			clip_mode=self.clip_mode,
			grad_bound=self.grad_bound,
			l2_reg=self.l2_reg,
			lr_init=self.lr_init,
			lr_dec_start=self.lr_dec_start,
			lr_dec_every=self.lr_dec_every,
			lr_dec_rate=self.lr_dec_rate,
			lr_cosine=self.lr_cosine,
			lr_max=self.lr_max,
			lr_min=self.lr_min,
			lr_T_0=self.lr_T_0,
			lr_T_mul=self.lr_T_mul,
			num_train_batches=self.num_train_batches,
			optim_algo=self.optim_algo,
			sync_replicas=self.sync_replicas,
			num_aggregate=self.num_aggregate,
			num_replicas=self.num_replicas)

	# override
	def _build_valid(self):
		if self.x_valid is not None:
			print("-" * 80)
			print("Build valid graph")
			logits = self._model(self.x_valid, False, reuse=True)
			self.valid_preds = tf.argmax(logits, axis=1)
			self.valid_preds = tf.to_int32(self.valid_preds)
			self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
			self.valid_acc = tf.to_int32(self.valid_acc)
			self.valid_acc = tf.reduce_sum(self.valid_acc)

	# override
	def _build_test(self):
		print("-" * 80)
		print("Build test graph")
		logits = self._model(self.x_test, False, reuse=True)
		self.test_preds = tf.argmax(logits, axis=1)
		self.test_preds = tf.to_int32(self.test_preds)
		self.test_acc = tf.equal(self.test_preds, self.y_test)
		self.test_acc = tf.to_int32(self.test_acc)
		self.test_acc = tf.reduce_sum(self.test_acc)

	# override
	def build_valid_rl(self, shuffle=False):
		print("-" * 80)
		print("Build valid graph on shuffled data")
		with tf.device("/cpu:0"):
			# shuffled valid data: for choosing validation model
			if not shuffle and self.data_format == "NCHW":
				self.images["valid_original"] = np.transpose(
					self.images["valid_original"], [0, 3, 1, 2])
			x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
					[self.images["valid_original"], self.labels["valid_original"]],
					batch_size=self.batch_size,
					capacity=25000,
					enqueue_many=True,
					min_after_dequeue=0,
					num_threads=16,
					seed=self.seed,
					allow_smaller_final_batch=True,
					)

			def _pre_process(x):
				inp_C=x.get_shape()[2]
				w_or_h=x.get_shape()[0]
				x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
				x = tf.random_crop(x, [w_or_h, w_or_h, inp_C], seed=self.seed)
				x = tf.image.random_flip_left_right(x, seed=self.seed)
				if self.data_format == "NCHW":
					x = tf.transpose(x, [2, 0, 1])
				return x
			
			if shuffle:
				x_valid_shuffle = tf.map_fn(
				_pre_process, x_valid_shuffle, back_prop=False)

		logits = self._model(x_valid_shuffle, False, reuse=True)
		valid_shuffle_preds = tf.argmax(logits, axis=1)
		valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
		self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
		self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
		self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)
		

	def connect_controller(self, controller_model):
		if self.fixed_arc is None:
			self.sample_arc = controller_model.sample_arc
		else:
			fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
			self.sample_arc = fixed_arc

		self._build_train()
		self._build_valid()
		self._build_test()

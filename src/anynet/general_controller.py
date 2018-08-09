import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class GeneralController(Controller):
	def __init__(self,
				search_for="both",
				search_whole_channels=False,
				num_layers=4,
				num_branches=6,
				out_filters=48,
				lstm_size=32,
				lstm_num_layers=2,
				lstm_keep_prob=1.0,
				tanh_constant=None,
				temperature=None,
				lr_init=1e-3,
				lr_dec_start=0,
				lr_dec_every=100,
				lr_dec_rate=0.9,
				l2_reg=0,
				entropy_weight=None,
				clip_mode=None,
				grad_bound=None,
				use_critic=False,
				bl_dec=0.999,
				optim_algo="adam",
				sync_replicas=False,
				num_aggregate=None,
				num_replicas=None,
				skip_target=0.8,
				skip_weight=0.5,
				name="controller",
				*args,
				**kwargs):

		print "-" * 80
		print "Building ConvController"

		self.search_for = search_for
		self.search_whole_channels = search_whole_channels
		self.num_layers = num_layers
		#moderated
		num_branches = num_layers
		self.num_branches = num_branches
		self.out_filters = out_filters

		self.lstm_size = lstm_size
		self.lstm_num_layers = lstm_num_layers 
		self.lstm_keep_prob = lstm_keep_prob
		self.tanh_constant = tanh_constant
		self.temperature = temperature
		self.lr_init = lr_init
		self.lr_dec_start = lr_dec_start
		self.lr_dec_every = lr_dec_every
		self.lr_dec_rate = lr_dec_rate
		self.l2_reg = l2_reg
		self.entropy_weight = entropy_weight
		self.clip_mode = clip_mode
		self.grad_bound = grad_bound
		self.use_critic = use_critic
		self.bl_dec = bl_dec

		self.skip_target = skip_target
		self.skip_weight = skip_weight

		self.optim_algo = optim_algo
		self.sync_replicas = sync_replicas
		self.num_aggregate = num_aggregate
		self.num_replicas = num_replicas
		self.name = name

		self._create_params()
		self._build_sampler()
		print (self.sample_arc)
		
	"""
	according to the main.py, lstm_num_layers=1 and lstm_size=64
	"""
	def _create_params(self):
		initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
		with tf.variable_scope(self.name, initializer=initializer):
			with tf.variable_scope("lstm"):
				self.w_lstm = []
				for layer_id in xrange(self.lstm_num_layers):
					with tf.variable_scope("layer_{}".format(layer_id)):
						w = tf.get_variable(
							"w", [2 * self.lstm_size, 4 * self.lstm_size])
						self.w_lstm.append(w)

			self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
			if self.search_whole_channels:
				with tf.variable_scope("emb"):
					self.w_emb = tf.get_variable(
						"w", [self.num_branches, self.lstm_size])
				with tf.variable_scope("softmax"):
					self.w_soft = tf.get_variable(
						"w", [self.lstm_size, self.num_branches])        #Just guess: self.num_branches decide the range of output of the controller LSTM
			else:
				raise ValueError("Just consider the situation when self.search_whole_channels=true")

			with tf.variable_scope("attention"):
				self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
				self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
				self.v_attn = tf.get_variable("v", [self.lstm_size, 1])

	def _build_sampler(self):
		"""Build the sampler ops and the log_prob ops."""

		print "-" * 80
		print "Build controller sampler"
		#anchors = []
		#anchors_w_1 = []

		arc_seq = []
		entropys = []
		log_probs = []
		#skip_count = []
		#skip_penaltys = []
		#initialize the first c and h as zero
		prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
				xrange(self.lstm_num_layers)]
		prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
				xrange(self.lstm_num_layers)]
		
		#shape(self.g_emb) = [1, self.lstm_size]
		inputs = self.g_emb           #the first input = self.g_emb
		#skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
		#				dtype=tf.float32)
		for layer_id in xrange(self.num_layers):
			if self.search_whole_channels:
				#the shapes of c and h are both [1,self.lstm_size]
				next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
				prev_c, prev_h = next_c, next_h
				#logit is the result of lstm after softmax
				logit = tf.matmul(next_h[-1], self.w_soft)
				if self.temperature is not None:
					logit /= self.temperature
				if self.tanh_constant is not None:
					logit = self.tanh_constant * tf.tanh(logit)
				if self.search_for == "macro" or self.search_for == "branch":
					branch_id = tf.multinomial(logit, 1)
					branch_id = tf.to_int32(branch_id)
					branch_id = tf.reshape(branch_id, [1])
				elif self.search_for == "connection":
					branch_id = tf.constant([0], dtype=tf.int32)
				else:
					raise ValueError("Unknown search_for {}".format(self.search_for))
				arc_seq.append(branch_id)
				log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logit, labels=branch_id)
				log_probs.append(log_prob)
				entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
				entropys.append(entropy)
				inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)
			else:
				raise ValueError("Just consider the situation when self.search_whole_channels=true")

			next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
			prev_c, prev_h = next_c, next_h
			"""
			if layer_id > 0:
				query = tf.concat(anchors_w_1, axis=0)
				query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
				query = tf.matmul(query, self.v_attn)
				logit = tf.concat([-query, query], axis=1)
				if self.temperature is not None:
					logit /= self.temperature
				if self.tanh_constant is not None:
					logit = self.tanh_constant * tf.tanh(logit)
				
				skip = tf.multinomial(logit, 1)
				skip = tf.to_int32(skip)
				skip = tf.reshape(skip, [layer_id])
				arc_seq.append(skip)
				
				skip_prob = tf.sigmoid(logit)
				kl = skip_prob * tf.log(skip_prob / skip_targets)
				kl = tf.reduce_sum(kl)
				skip_penaltys.append(kl)
				
				log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logit, labels=skip)
				log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))
				
				entropy = tf.stop_gradient(
					tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
				entropys.append(entropy)
				
				skip = tf.to_float(skip)
				skip = tf.reshape(skip, [1, layer_id])
				skip_count.append(tf.reduce_sum(skip))
				inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
				inputs /= (1.0 + tf.reduce_sum(skip))
			else:
				inputs = self.g_emb
			"""
			#anchors.append(next_h[-1])
			#anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

		arc_seq = tf.concat(arc_seq, axis=0)
		self.sample_arc = tf.reshape(arc_seq, [-1])

		entropys = tf.stack(entropys)
		self.sample_entropy = tf.reduce_sum(entropys)

		log_probs = tf.stack(log_probs)
		self.sample_log_prob = tf.reduce_sum(log_probs)

		#skip_count = tf.stack(skip_count)
		#self.skip_count = tf.reduce_sum(skip_count)

		#skip_penaltys = tf.stack(skip_penaltys)
		#self.skip_penaltys = tf.reduce_mean(skip_penaltys)

	def build_trainer(self, child_model):
		child_model.build_valid_rl()
		self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
						tf.to_float(child_model.batch_size))
		self.reward = self.valid_acc
		
		normalize = tf.to_float(self.num_layers * (self.num_layers - 1) / 2)
		#self.skip_rate = tf.to_float(self.skip_count) / normalize
		
		if self.entropy_weight is not None:
			self.reward += self.entropy_weight * self.sample_entropy
		
		self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
		self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
		baseline_update = tf.assign_sub(
			self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))
		
		with tf.control_dependencies([baseline_update]):
			self.reward = tf.identity(self.reward)
		
		self.loss = self.sample_log_prob * (self.reward - self.baseline)
		#if self.skip_weight is not None:
		#	self.loss += self.skip_weight * self.skip_penaltys
		
		self.train_step = tf.Variable(
			0, dtype=tf.int32, trainable=False, name="train_step")
		tf_variables = [var
			for var in tf.trainable_variables() if var.name.startswith(self.name)]
		print "-" * 80
		for var in tf_variables:
			print var

		self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
			self.loss,
			tf_variables,
			self.train_step,
			clip_mode=self.clip_mode,
			grad_bound=self.grad_bound,
			l2_reg=self.l2_reg,
			lr_init=self.lr_init,
			lr_dec_start=self.lr_dec_start,
			lr_dec_every=self.lr_dec_every,
			lr_dec_rate=self.lr_dec_rate,
			optim_algo=self.optim_algo,
			sync_replicas=self.sync_replicas,
			num_aggregate=self.num_aggregate,
			num_replicas=self.num_replicas)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np
from utils import preprocess

class VTranse(object):
	def __init__(self):
		self.predictions = {}
		self.losses = {}
		self.layers = {}
		self.feat_stride = [16, ]
		self.scope = 'vgg_16'
		self.POOLING_SIZE=7

	def create_graph(self, N_each_batch, index_sp, index_cls, num_classes, num_predicates,out_feature=500,train_feature=True):
		self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
		self.sbox = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.obox = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.sub_sp_info = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.ob_sp_info = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.rela_label = tf.placeholder(tf.int32, shape=[None,])
		self.keep_prob = tf.placeholder(tf.float32)
		self.index_sp = index_sp
		self.index_cls = index_cls
		self.num_classes = num_classes
		self.num_predicates = num_predicates
		self.N_each_batch = N_each_batch
		self.out_feature = out_feature
		self.concat_feature = tf.placeholder(tf.float32, shape=[None, out_feature * 2])

		# self.sub_fea = tf.placeholder(tf.float32, shape=[None, 500])
		# self.obj_fea = tf.placeholder(tf.float32, shape=[None, 500])
		# self.rela_label = tf.placeholder(tf.int32, shape=[None, ])

		if not train_feature:
			self.build_dete_network()#vgg_16
		self.build_rd_network(train_feature)
		self.add_rd_loss()

	def build_dete_network(self, is_training=True):
		net_conv = self.image_to_head(is_training)#vgg_16
		sub_pool5 = self.crop_pool_layer(net_conv, self.sbox, "sub_pool5")
		ob_pool5 = self.crop_pool_layer(net_conv, self.obox, "ob_pool5")
		sub_fc7 = self.head_to_tail(sub_pool5, is_training, reuse = False)#vgg_16
		ob_fc7 = self.head_to_tail(ob_pool5, is_training, reuse = True)#vgg_16

		with tf.variable_scope(self.scope, self.scope):
			# region classification
			sub_cls_prob, sub_cls_pred = self.region_classification(sub_fc7, is_training, reuse = False)
		with tf.variable_scope(self.scope, self.scope):
			# region classification
			ob_cls_prob, ob_cls_pred = self.region_classification(ob_fc7, is_training, reuse = True)

		self.predictions['sub_cls_prob'] = sub_cls_prob
		self.predictions['sub_cls_pred'] = sub_cls_pred
		self.predictions['ob_cls_prob'] = ob_cls_prob
		self.predictions['ob_cls_pred'] = ob_cls_pred
		self.layers['sub_pool5'] = sub_pool5
		self.layers['ob_pool5'] = ob_pool5
		# print('sub_pool5',sub_pool5)
		# print('ob_pool5',ob_pool5)
		self.layers['sub_fc7'] = sub_fc7
		self.layers['ob_fc7'] = ob_fc7

	def image_to_head(self, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			net = slim.repeat(self.image, 2, slim.conv2d, 64, [3, 3], 
				trainable=is_training, scope='conv1')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
				trainable=is_training, scope='conv2')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
				trainable=is_training, scope='conv3')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv4')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
			net_conv = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv5')
			self.layers['head'] = net_conv
			return net_conv

	def head_to_tail(self, pool5, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			pool5_flat = slim.flatten(pool5, scope='flatten')
			fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
			fc6 = slim.dropout(fc6, keep_prob=self.keep_prob, is_training=is_training,
					scope='dropout6')
			fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
			fc7 = slim.dropout(fc7, keep_prob=self.keep_prob, is_training=is_training,
					scope='dropout7')
			return fc7

	def crop_pool_layer(self, bottom, rois, name):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y \times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			n=tf.to_int32(rois.shape[0])
			batch_ids = tf.zeros([n,],dtype=tf.int32)
			# Get the normalized coordinates of bboxes
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
			x1 = tf.slice(rois, [0, 0], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 1], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 2], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 3], [-1, 1], name="y2") / height
			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [self.POOLING_SIZE*2, self.POOLING_SIZE*2], method='bilinear',
											 name="crops")
			pooling = max_pool(crops, 2, 2, 2, 2, name="max_pooling")
		return pooling

	def region_classification(self, fc7, is_training, reuse = False):
		cls_score = slim.fully_connected(fc7, self.num_classes, activation_fn=None, scope='cls_score', reuse=reuse)
		print("cls_score's shape: {0}".format(cls_score.get_shape()))
		cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
		cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

		return cls_prob, cls_pred

	def build_rd_network(self,train_feature):
		if not train_feature:
			sub_sp_info = self.sub_sp_info
			ob_sp_info = self.ob_sp_info
			sub_cls_prob = self.predictions['sub_cls_prob']
			ob_cls_prob = self.predictions['ob_cls_prob']
			sub_fc = self.layers['sub_fc7']
			ob_fc = self.layers['ob_fc7']
			if self.index_sp: #false
				sub_fc = tf.concat([sub_fc, sub_sp_info], axis = 1)
				ob_fc = tf.concat([ob_fc, ob_sp_info], axis = 1)
			if self.index_cls:#false
				sub_fc = tf.concat([sub_fc, sub_cls_prob], axis = 1)
				ob_fc = tf.concat([ob_fc, ob_cls_prob], axis = 1)
			sub_fc1 = slim.fully_connected(sub_fc, self.out_feature, activation_fn=tf.nn.relu, scope='RD_sub_fc1')  # out_feature:500
			ob_fc1 = slim.fully_connected(ob_fc, self.out_feature, activation_fn=tf.nn.relu, scope='RD_ob_fc1')
			self.layers['sub_fc1'] = sub_fc1
			self.layers['obj_fc1'] = ob_fc1
		#
		# dif_fc1 = ob_fc1 - sub_fc1 # here the ob_fc1 and sub_fc1 are the features of obj and sub
		# rela_score = slim.fully_connected(dif_fc1, self.num_predicates, activation_fn=None, scope='RD_fc2')

		# res_1 = slim.fully_connected(self.concat_feature, 128, activation_fn=tf.nn.relu, scope='RD_fc0')
		# res_1 = dropout(res_1, self.keep_prob)
		# rela_score = slim.fully_connected(res_1, self.num_predicates, activation_fn=None, scope='RD_fc2')

		# res_1 = slim.fully_connected(self.concat_feature, 256, activation_fn=tf.nn.relu, scope='RD_fc0')
		# res_1 = dropout(res_1, self.keep_prob)
		# res_1 = slim.fully_connected(res_1, 128, activation_fn=tf.nn.relu, scope='RD_fc1')
		# res_1 = dropout(res_1, self.keep_prob)
		# rela_score = slim.fully_connected(res_1, self.num_predicates, activation_fn=None, scope='RD_fc2')

		res_1 = slim.fully_connected(self.concat_feature, 512, activation_fn=None,scope='RD_fc0')  # ,weights_regularizer=slim.l2_regularizer(1e-4)
		res_1 = tf.nn.softplus(res_1)
		res_1 = dropout(res_1, self.keep_prob)
		res_1 = slim.fully_connected(res_1, 128, activation_fn=None,scope='RD_fc1')  # ,weights_regularizer=slim.l2_regularizer(1e-4)
		res_1 = tf.nn.softplus(res_1)
		res_1 = dropout(res_1, self.keep_prob)
		rela_score = slim.fully_connected(res_1, self.num_predicates, activation_fn=None,scope='RD_fc2')  # ,weights_regularizer=slim.l2_regularizer(1e-4)

		rela_prob = tf.nn.softmax(rela_score)
		self.layers['rela_score'] = rela_score
		self.layers['rela_prob'] = rela_prob

	def add_rd_loss(self):
		rela_score = self.layers['rela_score']
		rela_prob = self.layers['rela_prob']
		rela_label = self.rela_label
		rd_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
									labels = rela_label, logits = rela_score))
		self.losses['rd_loss'] = rd_loss

		acc_each = tf.nn.in_top_k(rela_score, rela_label, 1)
		self.losses['acc_each'] = acc_each
		self.losses['acc'] = tf.reduce_mean(tf.cast(acc_each, tf.float32))

		rela_pred = tf.argmax(rela_score, 1)
		self.predictions['rela_pred'] = rela_pred

		rela_max_prob = tf.reduce_max(rela_prob, 1)
		self.predictions['rela_max_prob'] = rela_max_prob

		self.predictions['rela_prob_all'] = rela_prob
		self.predictions['rela_score_all'] = rela_score

	def train_predicate(self, sess, roidb_use, RD_train):
		im, im_scale = preprocess.im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred']) / self.N_each_batch
		RD_loss = 0.0
		acc = 0.0

		for batch_id in range(np.int32(batch_num)):
			blob = preprocess.get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'],
						 self.rela_label: blob['rela'],
						 self.keep_prob: 0.5}
			_, losses, layers = sess.run([RD_train, self.losses, self.layers], feed_dict=feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']

		RD_loss = RD_loss / batch_num
		acc = acc / batch_num
		return RD_loss, acc

	def val_predicate(self, sess, roidb_use):
		im, im_scale = preprocess.im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		RD_loss = 0.0
		acc = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = preprocess.get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			losses = sess.run(self.losses, feed_dict = feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']

		RD_loss = RD_loss/batch_num
		acc = acc/batch_num
		return RD_loss, acc

	def extract_feature(self, sess, roidb_use):
		im, im_scale = preprocess.im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		pred_rela = np.zeros([len(roidb_use['index_pred']),])
		pred_rela_score = np.zeros([len(roidb_use['index_pred']),])
		real_length=len(roidb_use['rela_gt'])

		roidb_temp={}
		roidb_temp['img_path'] = roidb_use['image']
		roidb_temp['rela_labels'] = np.int32(roidb_use['rela_gt'])
		roidb_temp['sub_labels'] = np.int32(roidb_use['sub_gt'])
		roidb_temp['obj_labels'] = np.int32(roidb_use['obj_gt'])

		roidb_temp['sub_box'] = roidb_use['sub_box_gt']
		roidb_temp['obj_box'] = roidb_use['obj_box_gt']

		roidb_temp['sub_feas_500'] = []
		roidb_temp['obj_feas_500'] = []

		for batch_id in range(np.int32(batch_num)):
			# blob = get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			blob = preprocess.get_blob_pred_all(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			predictions,layers = sess.run([self.predictions,self.layers], feed_dict = feed_dict)

			if batch_id==np.int32(batch_num)-1:
				roidb_temp['sub_feas_500'].append(layers['sub_fc1'][0:real_length-self.N_each_batch*batch_id,:])
				roidb_temp['obj_feas_500'].append(layers['obj_fc1'][0:real_length-self.N_each_batch*batch_id,:])
			else:
				roidb_temp['sub_feas_500'].append(layers['sub_fc1'][0:self.N_each_batch,:])
				roidb_temp['obj_feas_500'].append(layers['obj_fc1'][0:self.N_each_batch,:])

			# pred_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_pred'][:]
			# pred_rela_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_max_prob'][:]

		N_rela = len(roidb_use['rela_gt'])
		# pred_rela = pred_rela[0:N_rela]
		# pred_rela_score = pred_rela_score[0:N_rela]

		assert real_length==len(roidb_use['rela_gt'])
		length_tmp=0
		for i in range(len(roidb_temp['sub_feas_500'])):
			length_tmp+=len(roidb_temp['sub_feas_500'][i])
		assert real_length==length_tmp
		return roidb_temp

	def train_predicate_fea_concate(self, sess, concat_fea, labels, RD_train,keep_prob):
		RD_loss = 0.0
		acc = 0.0
		feed_dict = {self.concat_feature: concat_fea, self.rela_label: labels, self.keep_prob: keep_prob}
		_, losses, layers = sess.run([RD_train, self.losses, self.layers], feed_dict=feed_dict)
		RD_loss = RD_loss + losses['rd_loss']
		acc = acc + losses['acc']
		return RD_loss, acc, losses['acc_each']

	def val_predicate_fea_concate(self, sess, concat_feature, labels):
		RD_loss = 0.0
		acc = 0.0
		feed_dict = {self.concat_feature: concat_feature, self.rela_label: labels, self.keep_prob: 1.0}
		losses = sess.run(self.losses, feed_dict=feed_dict)
		RD_loss = RD_loss + losses['rd_loss']
		acc = acc + losses['acc']
		return RD_loss, acc, losses['acc_each']

	def test_predicate(self, sess, roidb_use):
		assert roidb_use['sub_feas_500'].shape[0] == roidb_use['obj_feas_500'].shape[0] == \
			   roidb_use['rela_labels'].shape[0]
		feed_dict = {self.concat_feature:np.concatenate((roidb_use['sub_feas_500'], roidb_use['obj_feas_500']),axis=1),
					 self.rela_label: roidb_use['rela_labels'], self.keep_prob: 1}

		# feed_dict = {self.sub_fea: roidb_use['sub_feas_500'],self.obj_fea: roidb_use['obj_feas_500'],
		# 			self.rela_label: roidb_use['rela_labels'], self.keep_prob: 1}

		predictions = sess.run(self.predictions, feed_dict=feed_dict)
		pred_rela = predictions['rela_pred'][:]
		pred_rela_score = predictions['rela_max_prob'][:]
		rela_prob_all = predictions['rela_prob_all'][:]
		pred_rela_score_all = predictions['rela_score_all'][:]
		return pred_rela, pred_rela_score, rela_prob_all, pred_rela_score_all

def conv(x, h, w, K, s_y, s_x, name, relu = True, reuse=False, padding='SAME'):
	"""
	Args:
		x: input
		h: height of filter
		w: width of filter
		K: number of filters
		s_y: stride of height of filter
		s_x: stride of width of filter
	"""
	#c means the number of input channels
	c = int(x.get_shape()[-1])
	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[h,w,c,K])
		biases = tf.get_variable('biases', shape=[K])
		conv_value = tf.nn.conv2d(x, weights, strides = [1,s_y,s_x,1], padding = padding)
		add_baises_value = tf.reshape(tf.nn.bias_add(conv_value, biases), tf.shape(conv_value))
		if relu==True:
			relu_value = tf.nn.relu(add_baises_value, name=scope.name)
		else:
			relu_value = add_baises_value
		return relu_value

def fc(x,K,name,relu=True,reuse=False):
	"""
	Args:
		x: input
		K: the dimension of the output
	"""
	#c means the number of input channels
	c = int(x.get_shape()[1])
	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[c,K])
		biases = tf.get_variable('biases',shape=[K])
		relu_value = tf.nn.xw_plus_b(x,weights,biases,name = scope.name)
		if relu:
			result_value = tf.nn.relu(relu_value)
		else:
			result_value = relu_value
		return result_value

def max_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def avg_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.avg_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def leaky_relu(x, alpha):
	return tf.maximum(x, alpha * x)

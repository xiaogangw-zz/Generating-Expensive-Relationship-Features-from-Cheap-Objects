# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import itertools
import h5py
import os
import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'net'))
# import tflib as lib
# # import tflib.ops.linear

def w2v_encoder(dim_G,feature_1,keep_prob,reuse=False,name="Generator",training=False):
    with tf.variable_scope(name+'/w2v_encoder', reuse=reuse):
        inputs = feature_1
        inputs = slim.fully_connected(inputs, dim_G, activation_fn=tf.nn.leaky_relu)
        out=slim.dropout(inputs, keep_prob,is_training=training, scope='dropout1')
        out = slim.fully_connected(out, 1000, activation_fn=None)
        out=slim.dropout(out, keep_prob, is_training=training,scope='dropout2')
        return out

def ST_encoder(dim_G,feature_1,keep_prob,reuse=False,name="Generator",training=False):
    with tf.variable_scope(name+"/encoder", reuse=reuse):
        inputs = feature_1
        inputs = slim.fully_connected(inputs, dim_G, activation_fn=tf.nn.leaky_relu)
        inputs=slim.dropout(inputs, keep_prob,is_training=training, scope='dropout1')
        out = slim.fully_connected(inputs, 16, activation_fn=tf.nn.leaky_relu)
        out=slim.dropout(out, keep_prob,is_training=training, scope='dropout2')
        return out

def ST_decoder(dim_D,out_dim,feature_1,feature_2,keep_prob,wordvector=None,reuse=False,name='Generator',training=False):
    with tf.variable_scope(name+'/decoder', reuse=reuse):
        if wordvector==None:
            inputs = tf.concat([feature_1,feature_2], 1)
        else:
            inputs = tf.concat([feature_1,feature_2,wordvector], 1)
        inputs = slim.fully_connected(inputs, dim_D, activation_fn=tf.nn.leaky_relu)
        inputs=slim.dropout(inputs, keep_prob,is_training=training, scope='dropout1')
        out = slim.fully_connected(inputs, out_dim, activation_fn=tf.nn.leaky_relu)
        out=slim.dropout(out, keep_prob,is_training=training, scope='dropout2')
        return out

# def LeakyReLU(x, alpha=0.2):
#     return tf.maximum(alpha * x, x)
#
# def LeakyReLULayer(name, n_in, n_out, inputs):
#     output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
#     return LeakyReLU(output)

def netD(hidden_num_D,features1, n_layers=1,features2=None,  name="Discriminator", reuse=False):
    ### input is subject, object, object-subject ###
    if features2 is None:
        inputs = features1
    else:
        inputs=tf.concat([features1,features2],1)
    with tf.variable_scope(name, reuse=reuse):
        input_dim = inputs.get_shape().as_list()[-1]
        # n_layers = 3
        # output = LeakyReLULayer(name + 'Discriminator.Input', input_dim, hidden_num_D, inputs)
        output = slim.fully_connected(inputs, hidden_num_D, activation_fn=tf.nn.leaky_relu)
        for i in range(n_layers):
            # output = LeakyReLULayer(name + 'Discriminator.{}'.format(i), hidden_num_D, hidden_num_D / 2, output)
            output = slim.fully_connected(output, int(hidden_num_D / 2), activation_fn=tf.nn.leaky_relu)
            hidden_num_D = hidden_num_D / 2
        # output = lib.ops.linear.Linear(name + 'Discriminator.Out', hidden_num_D, 1, output)
        output = slim.fully_connected(output, 1, activation_fn=None)
        return tf.reshape(output, [-1])

def aux_classifier(features1, real_labels, num_predicates,keep_prob,name="ac_coder", reuse=False,training=False):
    with tf.variable_scope(name, reuse=reuse):
        inputs=features1
        # inputs = slim.fully_connected(inputs, 256, activation_fn=tf.nn.relu, scope='RD_fc0')
        # inputs = slim.dropout(inputs, keep_prob,is_training=training, scope='dropout1')
        # inputs = slim.fully_connected(inputs, 128, activation_fn=tf.nn.relu, scope='RD_fc1')
        # inputs = slim.dropout(inputs, keep_prob,is_training=training, scope='dropout2')
        rela_score = slim.fully_connected(inputs, num_predicates, activation_fn=None, scope='RD_fc2')
        out_rela_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_labels, logits=rela_score))
        return out_rela_loss #, out_acc, out_rela_pred, out_rela_max_prob

def cal_loss(errD_real,errD_fake):
    errD = tf.reduce_mean(errD_fake) - tf.reduce_mean(errD_real)
    errG = -tf.reduce_mean(errD_fake)
    return errD,errG

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def find_lowshot(path,gt_data,k):
    low_shot_ind=np.array(np.load(path+'/low_shot_ind_'+str(k)+'.npz')['roidb'][()]).astype(np.int32)
    low_shot_ind_v = low_shot_ind.view([('', low_shot_ind.dtype)] * low_shot_ind.shape[1]).ravel()
    gt_triplet=np.column_stack((gt_data['sub_label'],gt_data['pre_label'],gt_data['obj_label'])).astype(np.int32)
    gt_triplet_v=gt_triplet.view([('', gt_triplet.dtype)] * gt_triplet.shape[1]).ravel()
    gt_need_index=np.in1d(gt_triplet_v, low_shot_ind_v)
    return gt_need_index

def feed_data_centroids(input_data):
    analogies=input_data['good_centroids'].value
    centroids=input_data['centroids'].value
    labels=input_data['centroids_label'].value
    permuted_analogies = analogies[np.random.permutation(analogies.shape[0])]
    concatenated_centroids = np.concatenate(centroids, axis=0)
    data_num=permuted_analogies.shape[0]
    return data_num,permuted_analogies,concatenated_centroids,labels

def feed_pure_data(input_data):
    data_num=input_data['indexs'].shape[0]
    training_list=input_data['indexs'].value
    training_data=np.concatenate((input_data['sub_fea'],input_data['obj_fea']),axis=1)
    assert input_data['pre_label'].shape[0]==training_data.shape[0]
    return data_num,training_list,training_data



def linear_delta(input, output_dim, name=None, stddev=0.01):
    with tf.variable_scope(name or 'linear'):
        w_init = tf.random_normal_initializer(stddev=stddev)
        b_init = tf.constant_initializer(0.0)
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=w_init)
        b = tf.get_variable('b', [output_dim], initializer=b_init)
        return tf.matmul(input, w)+ b

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def ST_encoder_delta(dim_G, feature_1, keep_prob, reuse=False, name="Generator", training=False):
    with tf.variable_scope(name + "/encoder", reuse=reuse):
        inputs = feature_1
        inputs = tf.nn.dropout(inputs, keep_prob)
        inputs = linear_delta(inputs, dim_G, name='1')
        inputs = tf.nn.dropout(lrelu(inputs), keep_prob)
        out = linear_delta(inputs, 16, name='2')
        return out

def ST_decoder_delta(dim_D,out_dim,feature_1,feature_2,keep_prob,wordvector=None,reuse=False,name='Generator',training=False):
    with tf.variable_scope(name+'/decoder', reuse=reuse):
        if wordvector==None:
            inputs = tf.concat([feature_1,feature_2], 1)
        else:
            inputs = tf.concat([feature_1,feature_2,wordvector], 1)
        inputs = linear_delta(inputs, dim_D, name='1')
        inputs=tf.nn.dropout(lrelu(inputs), keep_prob)
        out = linear_delta(inputs, out_dim, name='2')
        return out
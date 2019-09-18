# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from net.vtranse_model import VTranse
import os
import argparse
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network for relationship recognition')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data', dest='data', type=str, default='VG')
    parser.add_argument('--lowshot_path', dest='lowshot_path', type=str, default='data_files/low_shot_files/vg/')
    parser.add_argument('--model_path', dest='model_path', type=str, default='')
    parser.add_argument('--lowshot_num', dest='lowshot_num', type=str, default='1') # 0 1 5 10 20
    parser.add_argument('--test_path', dest='test_path', type=str,default='data_files/vg/all_test_single_relation.h5')
    args = parser.parse_args()
    return args
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.test_path is not None:
    test_feature_all = h5py.File(args.test_path, 'r')
    N_val = test_feature_all['pre_label'].shape[0]

test_triplet = np.column_stack((test_feature_all['sub_label'], test_feature_all['pre_label'], test_feature_all['obj_label'])).astype(np.int32)
test_triplet_v = test_triplet.view([('', test_triplet.dtype)] * test_triplet.shape[1]).ravel()
low_shot_list = [0, 1, 5, 10, 20]
test_need_index = {}
# train_need_index = {}
for i in low_shot_list:
    low_shot_ind = np.array(np.load(args.lowshot_path + 'low_shot_ind_' + str(i) + '.npz')['roidb'][()]).astype(np.int32)
    low_shot_ind_v = low_shot_ind.view([('', low_shot_ind.dtype)] * low_shot_ind.shape[1]).ravel()
    test_need_index[str(i)] = np.in1d(test_triplet_v, low_shot_ind_v)
    print(np.where(test_need_index[str(i)])[0].shape) #, np.where(train_need_index[str(i)])[0].shape

if args.data == 'VG':
    N_cls = 201
    N_rela = 100
    N_each_batch = 128
    # lr_init = args.learning_rate
    N_each_batch_test = 4096
    # max_epoch = args.max_epoch
elif args.data == 'VRD':
    N_cls = 101
    N_rela = 70
    N_each_batch = 30 #int(30 / args.max_per_sample)
    # lr_init = args.learning_rate
    N_each_batch_test = 4096
    # max_epoch = args.max_epoch
vnet = VTranse()
vnet.create_graph(N_each_batch, False, False, N_cls, N_rela)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
total_var = tf.trainable_variables()
RD_var = [var for var in total_var if 'RD' in var.name]
saver = tf.train.Saver(var_list=RD_var)
with tf.Session(config=config) as sess:
    saver.restore(sess,args.model_path)
    acc_val = 0

    acc_val_lowshot = 0
    test_list_lowshot = np.arange(N_val)[test_need_index[str(args.lowshot_num)]]
    N_val_lowshot = test_list_lowshot.shape[0]
    # print(N_val_lowshot)
    for roidb_id in range(0, N_val_lowshot, N_each_batch_test):
        start_ind = roidb_id
        end_ind = min(start_ind + N_each_batch_test, N_val_lowshot)
        use_list = test_list_lowshot[start_ind:end_ind].tolist()
        labels = test_feature_all['pre_label'][use_list]
        gt_concat_fea = np.concatenate((test_feature_all['sub_fea'][use_list], test_feature_all['obj_fea'][use_list]), axis=1)
        rd_loss_temp, acc_temp, acc_each = vnet.val_predicate_fea_concate(sess, gt_concat_fea, labels)
        acc_val_lowshot += sum(acc_each)
    print("ls-" + str(args.lowshot_num) + " acc: {0}".format(acc_val_lowshot / N_val_lowshot))

test_feature_all.close()
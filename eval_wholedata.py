from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils.preprocess import *
from net.vtranse_model import VTranse
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network for relationship recognition')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data', dest='data', type=str, default='VG')
    parser.add_argument('--model_path', dest='model_path', type=str,default='')  # 'checkpoints/RelationModel/vg/wholedata/all_real/model/XXX'
    args = parser.parse_args()
    return args
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
exp_name=args.data # VG  VRD

if exp_name=='VG':
    N_cls = 201
    N_rela = 100
    N_each_batch = 128
    test_roidb = np.load('data_files/vg/test_vg_features_transfer.npz', encoding='latin1')['roidb'][()]
    model_path_2 = args.model_path #'checkpoints/RelationModel/vg/wholedata/all_real/model/XXX'

# N_train = len(train_roidb)
N_test = len(test_roidb)

## get the features ###
vnet = VTranse()
vnet.create_graph(N_each_batch, False, False, N_cls, N_rela)

total_var = tf.trainable_variables()
RD_var = [var for var in total_var if 'RD_fc' in var.name]
saver_RD_var = tf.train.Saver(var_list = RD_var)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver_RD_var.restore(sess, model_path_2)
    pred_roidb = []
    print(N_test)
    for roidb_id in range(N_test):
        if (roidb_id + 1)%10000 == 0:
            print(roidb_id + 1)
        roidb_use = test_roidb[roidb_id]
        if len(roidb_use['rela_labels']) == 0:
            pred_roidb.append({})
            continue
        pred_rela, pred_rela_score,pred_rela_prob_all,pred_rela_score_all = vnet.test_predicate(sess, roidb_use)    #test_predicate   gt_concat_fea,labels
        # print(pred_rela_prob_all.shape)
        # exit(0)
        pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
                            'pred_rela_prob_all': pred_rela_prob_all,'pred_rela_score_all': pred_rela_score_all,
                            'sub_box_dete': roidb_use['sub_box'], 'obj_box_dete': roidb_use['obj_box'],
                            'sub_dete': roidb_use['sub_labels'], 'obj_dete': roidb_use['obj_labels']}
        pred_roidb.append(pred_roidb_temp)

# ### save features ###
# roidb = {}
# roidb['pred_roidb'] = pred_roidb
# np.savez(save_path, roidb=roidb)
# print('save done')

R50, num_right50 = rela_recall_transfer(test_roidb, pred_roidb, 50)
R100, num_right100 = rela_recall_transfer(test_roidb, pred_roidb, 100)
print('R50: {0}, R100: {1}'.format(R50, R100))

# ### no graph constraint evaluation ###
# roidb_read = read_roidb(save_path)
# pred_roidb = roidb_read['pred_roidb']
# R50 ,R50zs= rela_recall_no_graph(test_roidb, pred_roidb, 50,exp_name)
# R100,R100zs = rela_recall_no_graph(test_roidb, pred_roidb, 100,exp_name)
# print('R50: {0}, R100: {1} R50zs: {2}, R100zs: {3}'.format(R50, R100,R50zs,R100zs))
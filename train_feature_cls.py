# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from net.vtranse_model import VTranse
from tensorboardX import summary
from tensorboardX import FileWriter
import operator
import os
import time
import argparse
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network for relationship recognition')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=50)
    parser.add_argument('--data', dest='data', type=str, default='VG')
    parser.add_argument('--mode', dest='mode', type=str, default='wholedata')   # wholedata  lowshot   
    parser.add_argument('--base_path', dest='base_path', type=str,default='checkpoints/RelationModel/vrd/')  #
    parser.add_argument('--output_path', dest='output_path', type=str, default='')  #
    parser.add_argument('--lowshot_path', dest='lowshot_path', type=str, default='data_files/low_shot_files/vg/')
    parser.add_argument('--train_path', dest='train_path', type=str,default='')
    parser.add_argument('--test_path', dest='test_path', type=str,default='data_files/vg/all_test_single_relation.h5')
    parser.add_argument('--lowshot_num', dest='lowshot_num', type=str, default='1')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--decay_epoch', type=int, default=20, help='decay epoch')
    parser.add_argument('--learning_rate_clip', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--eval_num', type=int, default=1000, help='evaluation interval')
    parser.add_argument('--print_num', type=int, default=5000, help='print interval')
    parser.add_argument('--keep_probability', dest='keep_probability', type=float, default=0.5)
    args = parser.parse_args()
    return args
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

lowshot_num=args.lowshot_num

if args.data == 'VG':
    N_cls = 201
    N_rela = 100
    N_each_batch = 128
    lr_init = args.learning_rate
    N_each_batch_test = 4096
    max_epoch = args.max_epoch
elif args.data == 'VRD':
    N_cls = 101
    N_rela = 70
    N_each_batch = 30
    lr_init = args.learning_rate
    N_each_batch_test = 4096
    max_epoch = args.max_epoch

vnet = VTranse()
vnet.create_graph(N_each_batch, False, False, N_cls, N_rela)

if args.mode == 'lowshot':
    base_path = args.base_path +args.mode+'/'+ lowshot_num + '/' + args.output_path
    model_path = base_path + '/model'
    log_path = base_path + '/log'
else:
    base_path = args.base_path  +args.mode+'/' + args.output_path
    model_path = base_path + '/model'
    log_path = base_path + '/log'

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
# exit(0)

with open(base_path + '/args.txt', 'w') as output_file:
    for x, y in vars(args).items():
        output_file.write("{} : {}\n".format(x, y))

summary_writer = FileWriter(log_path)

train_feature_all = h5py.File(args.train_path, 'r')
N_train_gt = train_feature_all['pre_label'].shape[0]
train_feature_use=train_feature_all['feature']
train_label_use=train_feature_all['pre_label']
N_train=N_train_gt
assert train_feature_use.shape[0] == train_label_use.shape[0]

if args.test_path is not None:
    test_feature_all = h5py.File(args.test_path, 'r')
    N_val = test_feature_all['pre_label'].shape[0]

def get_learning_rate(data_num,batch):
    learning_rate = tf.train.exponential_decay(
                        args.learning_rate,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        data_num//N_each_batch*args.decay_epoch,          # Decay step.
                        args.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, args.learning_rate_clip) # CLIP THE LEARNING RATE!
    return learning_rate

global_step = tf.Variable(0, name='global_step', trainable=False)
total_var = tf.trainable_variables()
RD_var = [var for var in total_var if 'RD' in var.name]
saver = tf.train.Saver(max_to_keep=None)
saver_res = tf.train.Saver(var_list=RD_var)

train_loss = vnet.losses['rd_loss']
learning_rate = get_learning_rate(N_train,global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
RD_train = optimizer.minimize(train_loss, global_step, var_list=RD_var)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if args.mode=='lowshot':
    ## find the lowshot indexes ###
    test_triplet = np.column_stack((test_feature_all['sub_label'], test_feature_all['pre_label'], test_feature_all['obj_label'])).astype(np.int32)
    test_triplet_v = test_triplet.view([('', test_triplet.dtype)] * test_triplet.shape[1]).ravel()
    low_shot_list = [0,1, 5, 10, 20]
    test_need_index = {}
    # train_need_index = {}
    for i in low_shot_list:
        low_shot_ind = np.array(np.load(args.lowshot_path + 'low_shot_ind_' + str(i) + '.npz')['roidb'][()]).astype(np.int32)
        low_shot_ind_v = low_shot_ind.view([('', low_shot_ind.dtype)] * low_shot_ind.shape[1]).ravel()
        test_need_index[str(i)] = np.in1d(test_triplet_v, low_shot_ind_v)
        # train_need_index[str(i)] = np.in1d(train_triplet_v, low_shot_ind_v)
        print(np.where(test_need_index[str(i)])[0].shape) #, np.where(train_need_index[str(i)])[0].shape

start_epoch = 0
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # restore previous model if there is one
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring previous model...")
        try:
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('_')[2]) + 1
            start_epoch=int(load_step/(N_train/N_each_batch))
            print(start_epoch)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from ", ckpt.model_checkpoint_path)
        except:
            start_epoch = 0
            print("Could not restore model")
            pass

    acc_val_all =-1
    acc_val_zeroshot_all =-1
    acc_val_lowshot_all =-1
    best_acc_val = 0
    best_acc_val_zeroshot = 0
    best_acc_val_lowshot = 0

    print('training examples:', N_train)
    train_list = np.arange(N_train)
    step=sess.run(global_step)
    for r in range(start_epoch, max_epoch):
        start_time = time.time()
        rd_loss = 0.0
        acc = 0.0

        train_list = np.random.permutation(train_list)
        for roidb_id in range(0, N_train, N_each_batch):
            step+=1
            start_ind = roidb_id
            end_ind = min(start_ind + N_each_batch, N_train)
            use_list = np.sort(train_list[start_ind:end_ind]).tolist()
            labels_all = train_label_use[use_list]
            fk_concat_fea_all = train_feature_use[use_list]

            rd_loss_temp, acc_temp, acc_each = vnet.train_predicate_fea_concate(sess, fk_concat_fea_all, labels_all,RD_train,args.keep_probability)
            rd_loss += rd_loss_temp
            acc += sum(acc_each)

            if (step + 1) % args.eval_num == 0:
                ###evaluation whole data###
                if args.mode == "wholedata":
                    rd_loss_val = 0.0
                    acc_val = 0

                    for val_id in range(0, N_val, N_each_batch_test):
                        start_ind = val_id
                        end_ind = min(start_ind + N_each_batch_test, N_val)
                        labels = test_feature_all['pre_label'][start_ind:end_ind]
                        gt_concat_fea = np.concatenate((test_feature_all['sub_fea'][start_ind:end_ind], test_feature_all['obj_fea'][start_ind:end_ind]),axis=1)
                        rd_loss_temp, acc_temp, acc_each = vnet.val_predicate_fea_concate(sess, gt_concat_fea, labels)
                        rd_loss_val = rd_loss_val + rd_loss_temp
                        acc_val += sum(acc_each)
                    print("whole-val: {0} rd_loss: {1}, acc: {2}, best_acc: {3}".format(step, rd_loss_val / N_val, acc_val / N_val,acc_val_all))

                    val_loss = summary.scalar('val_loss', rd_loss_val / N_val)
                    summary_writer.add_summary(val_loss, step)
                    val_accuracy = summary.scalar('val_acc', acc_val / N_val)
                    summary_writer.add_summary(val_accuracy, step)

                    if (acc_val / N_val) > acc_val_all:
                        save_path = model_path + '/' + args.data + '_vgg_' + format(int(step), '04')
                        saver.save(sess, save_path)
                        saver.export_meta_graph(save_path + '.meta')
                        acc_val_all=acc_val / N_val
                        best_acc_val=step

                ###evaluation lowshot dataset ###
                elif args.mode == "lowshot":
                    rd_loss_val_lowshot = 0.0
                    acc_val_lowshot = 0
                    test_list_lowshot = np.arange(N_val)[test_need_index[lowshot_num]]
                    N_val_lowshot = test_list_lowshot.shape[0]
                    # print(N_val_lowshot)
                    for roidb_id in range(0, N_val_lowshot, N_each_batch_test):
                        start_ind = roidb_id
                        end_ind = min(start_ind + N_each_batch_test, N_val_lowshot)
                        use_list = test_list_lowshot[start_ind:end_ind].tolist()
                        labels = test_feature_all['pre_label'][use_list]
                        gt_concat_fea = np.concatenate(
                            (test_feature_all['sub_fea'][use_list], test_feature_all['obj_fea'][use_list]), axis=1)

                        rd_loss_temp, acc_temp, acc_each = vnet.val_predicate_fea_concate(sess, gt_concat_fea, labels)
                        rd_loss_val_lowshot = rd_loss_val_lowshot + rd_loss_temp
                        acc_val_lowshot += sum(acc_each)
                    print("ls-" + lowshot_num + "-val: {0} rd_loss: {1}, acc: {2}, best_acc: {3}".format(step,rd_loss_val_lowshot / N_val_lowshot,acc_val_lowshot / N_val_lowshot,acc_val_lowshot_all))

                    val_loss_lowshot = summary.scalar('val_loss_lowshot', rd_loss_val_lowshot / N_val_lowshot)
                    summary_writer.add_summary(val_loss_lowshot, step)
                    val_accuracy_lowshot = summary.scalar('val_acc_lowshot', acc_val_lowshot / N_val_lowshot)
                    summary_writer.add_summary(val_accuracy_lowshot, step)

                    if (acc_val_lowshot / N_val_lowshot) > acc_val_lowshot_all:
                        save_path = model_path + '/' + args.data + '_vgg_' + format(int(step), '04')
                        saver.save(sess, save_path)
                        saver.export_meta_graph(save_path + '.meta')
                        acc_val_lowshot_all=acc_val_lowshot / N_val_lowshot
                        best_acc_val_lowshot = step

        loss = summary.scalar('loss', rd_loss / (N_train/N_each_batch))
        summary_writer.add_summary(loss, step)
        accuracy = summary.scalar('acc', acc / N_train)
        summary_writer.add_summary(accuracy, step)

    if args.mode == "lowshot":
        print('lowshot:', best_acc_val_lowshot , acc_val_lowshot_all)
        summary_writer.close()
    elif args.mode == "wholedata":
        print('wholedata:', best_acc_val , acc_val_all)
        summary_writer.close()

    train_feature_all.close()
    if args.test_path is not None:
        test_feature_all.close()

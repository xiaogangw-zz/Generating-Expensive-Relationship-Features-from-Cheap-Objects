# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from net.feature_model import *
import time
import shutil
from tensorboardX import summary
from tensorboardX import FileWriter
import operator
import os
import json
import argparse
import h5py
import random
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='Train a WGAN-GP network for feature generation')
    parser.add_argument('--dataset', dest='dataset', type=str, default='VG')
    parser.add_argument('--train_dir', dest='train_dir', type=str,default='data_files/vg/all_train_single_relation_random_5.h5')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='checkpoints/GenerationModel/vg/wholedata/')
    parser.add_argument('--test_setting', dest='test_setting', type=str, default='wholedata')   #  lowshot  wholedata
    parser.add_argument('--directory', dest='directory', type=str, default='main')
    parser.add_argument('--L1_weight', dest='L1_weight', type=float, default=100.0)  # 1000
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--num_predicates', dest='num_predicates', type=int, default=100)
    parser.add_argument('--ac_weight', dest='ac_weight', type=float, default=0.1)
    parser.add_argument('--training', dest='training', type=bool, default=False)  # True  False
    parser.add_argument('--output_dimension', dest='output_dimension', type=int, default=1000)
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=300)
    parser.add_argument('--input_dimension', dest='input_dimension', type=int, default=1000)
    parser.add_argument('--hidden_dimension', dest='hidden_dimension', type=int, default=8192)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)  # 512
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.8]')
    args = parser.parse_args()
    return args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
assert args.num_predicates==100 if args.dataset=='VG' else args.num_predicates==70
if args.dataset == 'VG':
    predicates = np.array(np.load('data_files/word_vectors/vg_predicates.npz')['roidb'][()])
# print(predicates)
# exit(0)
dim_G = dim_D = args.hidden_dimension
output_path=args.out_dir+args.directory+'_cls_weight_'+str(args.ac_weight)+'_L1_weight_'+str(args.L1_weight)+'/'
model_pth=output_path+'model/'
log_path=output_path+'log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(model_pth):
    os.makedirs(model_pth)

input_data = h5py.File(args.train_dir, 'r')
data_num = input_data['pre_label'].shape[0] # 803276

def get_learning_rate(data_num,batch):
    learning_rate = tf.train.exponential_decay(
                        args.learning_rate,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        data_num//args.batch_size*20,          # Decay step.
                        args.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-6) # CLIP THE LEARNING RATE!
    return learning_rate

start_epoch = 0
# summary_writer = FileWriter(log_path)
with open(output_path+'args.txt','w') as output_file:
    for x, y in vars(args).items():
        output_file.write("{} : {}\n".format(x, y))

def run(args):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_flag = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)
        feature_1 = tf.placeholder(tf.float32, [None, args.input_dimension], 'feature_1')
        feature_2 = tf.placeholder(tf.float32, [None, args.input_dimension], 'feature_2')
        feature_3 = tf.placeholder(tf.float32, [None, args.input_dimension], 'feature_3')
        real_labels=tf.placeholder(tf.int32, [None, ], 'real_labels')

        ### generated predicate features ###
        bottle_z = ST_encoder(dim_G, feature_1, keep_prob, reuse=False, training=train_flag)
        reconstruction = ST_decoder(dim_D,1000, bottle_z, feature_2, keep_prob, reuse=False, training=train_flag)
        errL1 = tf.reduce_mean(tf.losses.absolute_difference(reconstruction, feature_3, reduction=tf.losses.Reduction.NONE))
        if args.ac_weight > 0:
            ac_loss = aux_classifier(reconstruction, real_labels, args.num_predicates, keep_prob, reuse=False, training=train_flag)
        else:
            ac_loss=tf.zeros(1,dtype=tf.dtypes.float32)
        #errL1 = tf.reduce_mean(tf.abs(reconstruction - feature_3))

        errD_fake = netD(256, reconstruction, n_layers=0, reuse=False)
        errD_real = netD(256, feature_3, n_layers=0, reuse=True)

        # cost functions
        errD = tf.reduce_mean(errD_fake) - tf.reduce_mean(errD_real)
        errG = -tf.reduce_mean(errD_fake)
        if args.ac_weight > 0:
            errG_total = errG + errL1 * args.L1_weight + args.ac_weight * ac_loss
        else:
            errG_total = errG + errL1 * args.L1_weight

        # gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = feature_3 * (1 - epsilon) + epsilon * reconstruction
        d_hat = netD(256,x_hat, n_layers=0, reuse=True)
        gradients = tf.gradients(d_hat, x_hat)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10 * tf.reduce_mean((slopes - 1.0) ** 2)
        errD_total = errD + gradient_penalty

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Generator' in var.name]
        learning_rate = get_learning_rate(data_num, global_step)
        G_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(errG_total,global_step,var_list=g_vars)
        D_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(errD_total,global_step,var_list=d_vars)

        ops = {'D_train_op': D_train_op,
               'G_train_op': G_train_op,
               'feature_1': feature_1,
               'feature_2': feature_2,
               'feature_3': feature_3,
               'keep_prob': keep_prob,
               'real_labels': real_labels,
               'train_flag': train_flag,
               'errD': errD,
               'errG': errG,
               'errL1': errL1,
               'ac_loss': ac_loss,
               'reconstruction': reconstruction}

        saver = tf.train.Saver(max_to_keep=None)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        ### make gpu memory grow according to needed ###
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init)
        summary_writer = FileWriter(log_path, graph=tf.get_default_graph())

        # tf.add_to_collection('train_op', train_op)
        tf.add_to_collection('G_train_op', G_train_op)
        tf.add_to_collection('D_train_op', D_train_op)

        start_epoch=0
        if args.training:
            # restore previous model if there is one
            ckpt = tf.train.get_checkpoint_state(model_pth)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring previous model...")
                try:
                    start_epoch = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1]) + 1
                    print(start_epoch)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Model restored")
                except:
                    print("Could not restore model")
                    pass

            ########################################### training portion
            for epoch in range(start_epoch,args.max_epoch):
                start = time.time()
                train_loss_d, train_loss_g, train_loss_L1Loss,train_loss_acLoss = train_one_epoch(sess, input_data, ops, args)
                print('epoch:', epoch, 'D loss:', train_loss_d.avg, 'G_loss:', train_loss_g.avg, 'L1:',train_loss_L1Loss.avg,'AC:',train_loss_acLoss.avg, 'time:', time.time() - start)

                summary_D = summary.scalar('D_loss', train_loss_d.avg)
                summary_writer.add_summary(summary_D, epoch)
                summary_G = summary.scalar('G_loss', train_loss_g.avg)
                summary_writer.add_summary(summary_G, epoch)
                summary_G_L1 = summary.scalar('G_L1', train_loss_L1Loss.avg)
                summary_writer.add_summary(summary_G_L1, epoch)
                summary_AC = summary.scalar('G_AC', train_loss_acLoss.avg)
                summary_writer.add_summary(summary_AC, epoch)
                if (epoch + 1) % 10 == 0:
                    print('save model')
                    if not os.path.exists(model_pth):
                        os.makedirs(model_pth)
                    saver.save(sess, model_pth + 'checkpoint-' + str(epoch))
                    saver.export_meta_graph(model_pth + 'checkpoint-' + str(epoch) + '.meta')
        else:
            print('evaluation')
            ckpt = tf.train.get_checkpoint_state(model_pth)
            try:
                epoch = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored")
            except:
                print("Could not restore model")
                exit(0)
                pass

            ### generate whole data###
            if args.test_setting == 'wholedata':
                print('generate whole data:', epoch)
                generate_wholedata(sess, input_data, ops, epoch)
            ### generate lowshot vrd data###
            elif args.test_setting == 'lowshot':
                print('generate lowshot data:', epoch)
                generate_lowshot(sess, input_data, ops, args, epoch)
    input_data.close()

def train_one_epoch(sess, input_data, ops, args):
    train_loss_d = AverageMeter()
    train_loss_g = AverageMeter()
    train_loss_L1Loss = AverageMeter()
    train_loss_acLoss = AverageMeter()

    ran_sub_fea=input_data['ran_sub_fea'][()]
    ran_obj_fea = input_data['ran_obj_fea'][()]
    sub_fea=input_data['sub_fea'][()]
    obj_fea=input_data['obj_fea'][()]
    pre_label=input_data['pre_label'][()]
    ref_ind = input_data['ref_ind'][()]
    data_num=np.where(ref_ind>-1)[0].shape[0]
    data_list_ind = np.where(ref_ind>-1)[0]
    data_list=np.arange(input_data['pre_label'].shape[0])[data_list_ind]
    # data_num = len(whole_ind) - len(oneshot_ind)
    # data_list = list(set(range(data_num)) - set(oneshot_ind))

    #data_list = np.arange(data_num)
    np.random.shuffle(data_list)

    for i in range(0, data_num, args.batch_size):
        start_ind = i
        end_ind = min(start_ind + args.batch_size, data_num)
        id_3 = data_list[start_ind:end_ind] #np.sort(data_list[start_ind:end_ind])
        id_1 = id_3 * 5+1
        id_2 = ref_ind[id_3]

        feature_1_input = np.concatenate((ran_sub_fea[id_1],ran_obj_fea[id_1]), axis=1)
        feature_2_input = np.concatenate((sub_fea[id_2], obj_fea[id_2]), axis=1)
        feature_3_input = np.concatenate((sub_fea[id_3], obj_fea[id_3]), axis=1)
        pre_labels = pre_label[id_3]
        assert feature_1_input.shape == feature_2_input.shape == feature_3_input.shape

        for critic_itr in range(5):
            sess.run(ops['D_train_op'], feed_dict={ops['feature_1']: feature_1_input,
                                                   ops['feature_2']: feature_2_input,
                                                   ops['feature_3']: feature_3_input,
                                                   ops['keep_prob']: 0.5,
                                                   ops['real_labels']: pre_labels,
                                                   ops['train_flag']: True})
        _, D_loss, G_loss, G_L1,AC_loss = sess.run([ops['G_train_op'], ops['errD'], ops['errG'], ops['errL1'], ops['ac_loss']],
                                           feed_dict={ops['feature_1']: feature_1_input,
                                                      ops['feature_2']: feature_2_input,
                                                      ops['feature_3']: feature_3_input,
                                                      ops['keep_prob']: 0.5,
                                                      ops['real_labels']: pre_labels,
                                                      ops['train_flag']: True})
        train_loss_d.update(D_loss, args.batch_size)
        train_loss_g.update(G_loss, args.batch_size)
        train_loss_L1Loss.update(G_L1, args.batch_size)
        train_loss_acLoss.update(AC_loss, args.batch_size)

    return train_loss_d,train_loss_g,train_loss_L1Loss,train_loss_acLoss

def generate_wholedata(sess, input_data, ops, epoch):
    results_path = output_path + 'wholedata/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    data_list = np.arange(data_num)
    batch_size = 512

    ran_sub_fea = input_data['ran_sub_fea']
    ran_obj_fea = input_data['ran_obj_fea']
    sub_fea = input_data['sub_fea']
    obj_fea = input_data['obj_fea']
    pre_label = input_data['pre_label']

    # if os.path.exists(results_path + 'wholedata_output_' + str(epoch) + '_1.h5'):
    #     return

    feature_save = file.create_dataset('feature', (data_num*2, 1000), 'f', compression='gzip', chunks=(1, 1000))
    pre_label_save = file.create_dataset('pre_label', (data_num*2,), 'i', compression='gzip', chunks=(1,))
    feature_save[-data_num:] = np.concatenate((sub_fea, obj_fea), axis=1)
    pre_label_save[-data_num:] = pre_label

    for i in range(0,data_num,batch_size):
        start_ind = i
        end_ind = min(start_ind + batch_size, data_num)
        id_3 = data_list[start_ind:end_ind]
        id_2 = id_3 * 5
        id_1 = id_2 + 2
        feature_1_input = np.concatenate((ran_sub_fea[id_1.tolist()], ran_obj_fea[id_1.tolist()]), axis=1)
        feature_2_input = np.concatenate((sub_fea[id_3.tolist()], obj_fea[id_3.tolist()]), axis=1)
        pre_labels = pre_label[id_3.tolist()]
        assert feature_1_input.shape == feature_2_input.shape

        output = sess.run(ops['reconstruction'], feed_dict={ops['feature_1']: feature_1_input,
                                                            ops['feature_2']: feature_2_input,
                                                            ops['keep_prob']: 1.0,
                                                            ops['train_flag']: False})
        feature_save[start_ind:end_ind] = output
        pre_label_save[start_ind:end_ind] = pre_labels
    file.close()

def generate_lowshot(sess, input_data, ops, args,epoch):
    ### generate lowshot vg data ###
    results_path = output_path + 'lowshot/'
    batch_size = 64
    for shot_num in [1, 5, 10, 20]:
        gt_data = h5py.File('data_files/vg/lowshot_data_' + str(shot_num) + '_same_pre.h5', 'r')
        data_num_ori = gt_data['pre_label'].shape[0]
        gt_sub_fea = gt_data['sub_fea'] #np.repeat(gt_data['sub_fea'], 30, axis=0)
        gt_obj_fea = gt_data['obj_fea'] #np.repeat(gt_data['obj_fea'], 30, axis=0)
        gt_pre_label = gt_data['pre_label'] #.value
        # print(gt_pre_label.shape)
        # gt_pre_label = np.repeat(gt_pre_label, 30, axis=0)
        gt_ran_sub_fea = gt_data['ran_sub_fea']
        gt_ran_obj_fea = gt_data['ran_obj_fea']
        # print(gt_pre_label.shape, gt_ran_obj_fea.shape)
        # assert gt_pre_label.shape[0] == gt_ran_obj_fea.shape[0]
        data_list = np.arange(data_num_ori)
        ### generate 1024 samples ###
        for need_num in range(15, 19, 5):
            save_path = results_path + str(shot_num) + '/' + str(epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # save_path = results_path + str(shot_num) + '/'
            file = h5py.File(save_path + '/' + 'lowshot_output_' + str(shot_num) + '_' + str(need_num) + '.h5', 'w')
            feature_save = file.create_dataset('feature', (data_num_ori * need_num+data_num, 1000), 'f', compression='gzip', chunks=(1, 1000))
            pre_label_save = file.create_dataset('pre_label', (data_num_ori * need_num+data_num,), 'i', compression='gzip', chunks=(1,))
            feature_save[-data_num:] = np.concatenate((input_data['sub_fea'], input_data['obj_fea']), axis=1)
            pre_label_save[-data_num:] = input_data['pre_label']

            for val_ind in range(0, data_num_ori, batch_size):
                start_ind = val_ind
                end_ind = min(start_ind + batch_size, data_num_ori)
                id_3 = data_list[start_ind:end_ind]
                id_1 = id_3 * 30

                base_list_tmp = id_1
                for i in range(1, need_num):
                    tmp = base_list_tmp + i
                    id_1 = np.concatenate((id_1, tmp), axis=0)
                id_1 = np.sort(id_1)

                feature_1_input = np.concatenate((gt_ran_sub_fea[id_1.tolist()], gt_ran_obj_fea[id_1.tolist()]), axis=1)
                # gt_id_use=test_list_gt[id_use]
                feature_2_input = np.concatenate((gt_sub_fea[id_3.tolist()], gt_obj_fea[id_3.tolist()]), axis=1)
                feature_2_input = np.repeat(feature_2_input, need_num, axis=0)
                out_label = gt_pre_label[id_3.tolist()]
                out_label = np.repeat(out_label, need_num, axis=0)

                output = sess.run(ops['reconstruction'], feed_dict={ops['feature_1']: feature_1_input,
                                                                    ops['feature_2']: feature_2_input,
                                                                    ops['keep_prob']: 1.0,
                                                                    ops['train_flag']: False})

                feature_save[start_ind * need_num:end_ind * need_num] = output
                pre_label_save[start_ind * need_num:end_ind * need_num] = out_label
            file.close()
        gt_data.close()

if __name__ == "__main__":
    run(args)
import numpy as np
import os
import h5py

def transfer_path(npz_path):
    roidb_read = np.load(npz_path, allow_pickle=True)['roidb'][()]
    train_roidb = roidb_read['train_roidb']
    test_roidb = roidb_read['test_roidb']
    for i in range(len(train_roidb)):
        img_path=train_roidb[i]['image'].encode('utf8')
        img_path=img_path.replace("/home/DeepStore/wxg/vtrans_modify/vtranse_with_data/", "")
        train_roidb[i]['image']=img_path
    roidb_read['train_roidb']= train_roidb
    test_roidb = roidb_read['test_roidb']
    for i in range(len(test_roidb)):
        img_path=test_roidb[i]['image'].encode('utf8')
        img_path=img_path.replace("/home/DeepStore/wxg/vtrans_modify/vtranse_with_data/", "")
        test_roidb[i]['image']=img_path
    roidb_read['test_roidb'] = test_roidb
    np.savez('data_files/vg_roidb_utf8.npz', roidb=roidb_read)
    roidb_read = np.load('data_files/vg_roidb_utf8.npz',allow_pickle=True)['roidb'][()]
    train_roidb = roidb_read['train_roidb']
    test_roidb = roidb_read['test_roidb']

def npz2h5(path):
    train_feature_base = np.load(path, encoding='latin1')['roidb'][()]
    ran_sub_fea = np.asarray(train_feature_base['ran_sub_fea'])
    ran_obj_fea = np.asarray(train_feature_base['ran_obj_fea'])
    # print(data.shape)
    # exit(0)
    file = h5py.File('../data_files/vg/test_zeroshot_random_50.h5', 'w')
    file.create_dataset('sub_fea', data=train_feature_base['sub_fea'])
    file.create_dataset('obj_fea', data=train_feature_base['obj_fea'])
    file.create_dataset('ran_sub_fea', data=ran_sub_fea.reshape(-1,500))
    file.create_dataset('ran_obj_fea', data=ran_obj_fea.reshape(-1,500))
    file.create_dataset('sub_label', data=train_feature_base['sub_label'])
    file.create_dataset('obj_label', data=train_feature_base['obj_label'])
    file.create_dataset('pre_label', data=train_feature_base['pre_label'])
    file.close()

def vg_dat_selection(sub_path,obj_path,rela_pah,all_path):
    rela_ids = np.load(rela_pah, encoding='latin1')['roidb'][()]
    sub_ids = np.load(sub_path, encoding='latin1')['roidb'][()]
    obj_ids = np.load(obj_path, encoding='latin1')['roidb'][()]
    num=5
    # file = h5py.File(all_path, 'r+')
    # ran_sub_fea_ids=file['ran_sub_fea_ids']
    # ran_obj_fea_ids = file['ran_obj_fea_ids']
    # ran_sub_fea_ids = file.create_dataset('ran_sub_fea_ids', (len(rela_ids['index']),num), 'i', compression='gzip', chunks=(1,num))
    # ran_obj_fea_ids = file.create_dataset('ran_obj_fea_ids', (len(rela_ids['index']),num), 'i', compression='gzip',chunks=(1,num))

    minimium=100
    for i in range(len(rela_ids['index'])):
        print(i)
        if minimium>sub_ids['index'][i].shape[0] or minimium>obj_ids['index'][i].shape[0]:
            minimium=min(sub_ids['index'][i].shape[0],minimium>obj_ids['index'][i].shape[0])
    print(minimium)
        # if (i +1)%1000==0:
        #     print(i)
        # if rela_ids['index'][i].shape[0]<num:
        #     # print(rela_ids['index'][i], sub_ids['index'][i][0:num - rela_ids['index'][i].shape[0]])
        #     ran_sub_fea_ids[i]=np.concatenate((rela_ids['index'][i],sub_ids['index'][i][0:num-rela_ids['index'][i].shape[0]]))
        #
        #     # print(ran_sub_fea_ids[i])
        #     # exit(0)
        #     ran_obj_fea_ids[i] = np.concatenate((rela_ids['index'][i], obj_ids['index'][i][0:num - rela_ids['index'][i].shape[0]]))
        # else:
        #     ran_sub_fea_ids[i]=rela_ids['index'][i][0:num]
        #     ran_obj_fea_ids[i] = rela_ids['index'][i][0:num]
    # file.close()


npz2h5('/mnt/data2/wxg/ObjectRelationshipDetection/vtranse_stgan/dataset/vg/whole_data/NPZ_features/random_features/test_zeroshot_random_50.npz')

# sub_list=h5py.File('/mnt/data2/wxg/ObjectRelationshipDetection/vtranse_stgan/Generating-Expensive-Relationship-Features-from-Cheap-Objects/data_files/vg/all_train_same_pre_obj_indexs.h5','r')
# sub_list=sub_list['indexes'].value.astype(np.int32

# vg_dat_selection('../data_files/vg/train_vrd_all_same_PreSub_ind.npz',\
#                  '../data_files/vg/train_vrd_all_same_PreObj_ind.npz',\
#                  '../data_files/vg/train_vrd_all_same_PreSubObj_ind.npz',\
#                  '../data_files/vg/all_train_single_relation.h5')

# inputa=np.random.randn(3,5,2)
# print(inputa)
# print(inputa.reshape(-1,2))
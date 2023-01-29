from turtle import color
import numpy as np
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import math 
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import math

def visual_svd(s_list, method_list):
    x = np.arange(0,512)
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    plt.clf()
    for i in [5, 3, 6, 1, 2, 0, 4]:
        s_list[i] = s_list[i] / torch.sum(s_list[i])
        if method_list[i] == 'V-adv':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9, color = 'purple')
            continue
        if method_list[i] == 'V-new-drop':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.65, color='purple')
            continue
        if method_list[i] == 'V-trans':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.4, color='purple')
            continue
        # if method_list[i] == 'Joint':
        #     plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9, color='red')
            # continue
        plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9)

    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Log Singular Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.axvline(x = 50, linestyle='--', color = 'r', linewidth=1.5)
    ax = plt.gca()
    bwith = 1.
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    plt.tick_params(labelsize=14)
    plt.savefig('./traing_grad_svd.pdf')
    plt.show

def compute_variance(class_feat, m_feature):
    RMSSTD = 0
    for i_ter in range(class_feat.shape[0]):
        RMSSTD += 1 - torch.cosine_similarity(m_feature[0], class_feat[i_ter], dim=0)
    RMSSTD = RMSSTD / (class_feat.shape[0])
    return RMSSTD

def compute_angular(class_feat, m_feature):
    RMSSTD = 0
    for i_ter in range(class_feat.shape[0]):
        cos_sim = torch.cosine_similarity(m_feature[0], class_feat[i_ter], dim=0)
        angular_ = torch.arccos(cos_sim)
        RMSSTD += angular_
    RMSSTD = RMSSTD / (class_feat.shape[0])
    return RMSSTD


def trans_feat(feat, label, ori_feat, new_feat, new_label):
    feat_all = []
    label_all = []
    new_feat_all = []
    new_label_all = []
    ori_feat_all = []
    for len_i in range(label.shape[0]):
        if len(feat[len_i]) == len(feat[2]):
            feat_all.append(feat[len_i])
            label_all.append(label[len_i])
            ori_feat_all.append(ori_feat[len_i])
            new_feat_all.append(new_feat[len_i])
            new_label_all.append(new_label[len_i])
    feat = torch.tensor(list(feat_all))
    label = torch.tensor(list(label_all))
    ori_feat = torch.tensor(list(ori_feat_all))
    new_feat = torch.tensor(list(new_feat_all))
    new_label = torch.tensor(list(new_label_all))

    new_feat = torch.reshape(new_feat, (new_feat.shape[0] * new_feat.shape[1], new_feat.shape[2]))
    new_label = torch.reshape(new_label, (new_label.shape[0] * new_label.shape[1],))
    feat = torch.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2]))
    ori_feat = torch.reshape(ori_feat, (ori_feat.shape[0] * ori_feat.shape[1], ori_feat.shape[2]))
    label = torch.reshape(label, (label.shape[0] * label.shape[1],))
    return feat, label, ori_feat, new_feat, new_label

EPS = 1E-20
current_task = '5'
target_type = 'target_buf_labels'
gamma_loss = '1.0'
noise_type = 'noise'
method2 = 'gaussian'
c_theta = '45.0'
para_scale = '1.0'
feat_dict = {
'adv'           :'buff_featurte_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'buff_featurte_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'buff_featurte_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'buff_featurte_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'buff_featurte_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'buff_featurte_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'buff_featurte_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'buff_featurte_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'buff_featurte_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'buff_featurte_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}

new_feat_dict = {
'adv'           :'new_featurte_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'new_featurte_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'new_featurte_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'new_featurte_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'new_featurte_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'new_featurte_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'new_featurte_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'new_featurte_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'new_featurte_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'new_featurte_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}

new_label_dict = {
'adv'           :'new_labels_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'new_labels_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'new_labels_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'new_labels_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'new_labels_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'new_labels_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'new_labels_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'new_labels_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'new_labels_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'new_labels_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}

noise_dict = {
'adv'           :'buff_noise_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'buff_noise_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'buff_noise_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'buff_noise_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'buff_noise_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'buff_noise_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'buff_noise_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'buff_noise_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'buff_noise_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'buff_noise_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}

label_dict = {
'adv'           :'buff_labels_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'buff_labels_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'buff_labels_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'buff_labels_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'buff_labels_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'buff_labels_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'buff_labels_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'buff_labels_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'buff_labels_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'buff_labels_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}

ori_dict = {
'adv'           :'ori_buffer_feat_task_2_new_labels_50.0_adv_vmf_45.0_1.0.npy',
'class_mean'    :'ori_buffer_feat_task_2_new_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'ori_buffer_feat_task_2_new_labels_1.0_noise_drop_new_45.0_1.0.npy',
'er_500'        :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_500_45.0_0.0.npy',
'gaussian'      :'ori_buffer_feat_task_2_new_labels_1.0_noise_gaussian_45.0_1.0.npy',
'joint'         :'ori_buffer_feat_task_2_new_labels_1.0_noise_joint_45.0_0.0.npy',
'vmf_1'         :'ori_buffer_feat_task_2_new_labels_1.0_noise_vmf_45.0_1.0.npy',
'vmf_1000'      :'ori_buffer_feat_task_2_new_labels_1000.0_noise_vmf_45.0_1.0.npy',
'ace'           :'ori_buffer_task_2_new_labels_1.0_noise_ace_45.0_0.0.npy',
'pp'            :'ori_buffer_task_2_new_labels_1.0_noise_pp_45.0_0.0.npy'
}


s_list = []
method_list = [
'adv'           ,         
'class_mean'    ,
'drop_new'      ,
'er_500'        ,
'gaussian'      ,
'joint'         ,
'vmf_1'         ,
'vmf_1000'   
]
# method_list = [
# 'pp'            ,             #0.5543680574357451 0.6701228379876824  
# 'er_500'    ,                   #0.5257 0.6724
# 'ace'      ,                  #0.5602325831728716 0.6352455812074242
# 'joint'        ,                #0.7182     0.7146
# 'gaussian'      ,               #0.8950     0.6853
# 'adv'         ,                 #0.6412     0.6855              
# ]

method_list = [
'class_mean'            ,             #0.5543680574357451 0.6701228379876824  
'drop_new'    ,                   #0.5257 0.6724
'vmf_1'      ,                  #0.5602325831728716 0.6352455812074242          
]
ALL_RMSSTD_list = []
for idx, method2 in enumerate(method_list):
    name = feat_dict[method2]
    name = './output_variance/' + name
    buf_feat = np.load(name, allow_pickle=True)

    name = new_feat_dict[method2]
    name = './output_variance/' + name
    new_feat = np.load(name, allow_pickle=True)

    name = label_dict[method2]
    name = './output_variance/' + name
    buf_label = np.load(name, allow_pickle=True)

    name = new_label_dict[method2]
    name = './output_variance/' + name
    new_label = np.load(name, allow_pickle=True)

    feat, label, ori_feat, new_feat, new_label = trans_feat(buf_feat, buf_label, new_feat, new_feat, new_label)
    
    epoch = 1
    num_per_epoch = int(feat.shape[0]/epoch)
    feat_all = torch.cat((feat[-num_per_epoch:], new_feat[-num_per_epoch:]), 0)
    label_all = torch.cat((label[-num_per_epoch:], new_label[-num_per_epoch:]), 0)

    RMSSTD_list = []
    for i_class in range(0,100):
        mask = label_all == i_class
        class_feat = feat_all[mask] 
        class_feat = F.normalize(class_feat, dim=1, p=2)
        m_feature = torch.mean(class_feat, dim=0).unsqueeze(0)
        m_feature = F.normalize(m_feature, dim=1, p=2)

        if class_feat.shape[0] != 0:
            RMSSTD = compute_angular(class_feat, m_feature)
            # RMSSTD = compute_variance(class_feat, m_feature)
            RMSSTD_list.append(RMSSTD.item())
        else:
            RMSSTD_list.append((RMSSTD_list[-1]+RMSSTD_list[-2])/2)
    ALL_RMSSTD_list.append(RMSSTD_list)
    print(sum(RMSSTD_list[:50])/50, sum(RMSSTD_list[50:])/50)

# import numpy as np
# ALL_RMSSTD_list=np.array(ALL_RMSSTD_list)
# np.save('pp_variance.npy',ALL_RMSSTD_list)







# all_1 = torch.from_numpy(np.load('all_method_variance.npy'))
# all_2 = torch.from_numpy(np.load('ace_pp_variance.npy'))
# all_3 = torch.from_numpy(np.load('pp_variance.npy'))
# all_2[1, :] = all_3

# all_RMSSTD = torch.cat((all_1, all_2), 0)
# print(all_RMSSTD.shape)

# method_list = [
# 'V-adv'           ,         
# 'V-trans'    ,
# 'V-new-drop'      ,
# 'ER'        ,
# 'V-gaussian'      ,
# 'Joint'         ,
# 'V-vmf'         ,
# 'V-vmf-1000'      ,   
# 'ER-ACE'           ,    
# 'DER++'   
# ]

# all_RMSSTD = torch.load('all_method_variance.pt')
# # all_RMSSTD[5][:50] = all_RMSSTD[5][:50] + 0.035
# # torch.save(all_RMSSTD, 'all_method_variance.pt')

# def visual_variance(s_list, method_list):
#     x = np.arange(0,100)
#     from matplotlib.ticker import MaxNLocator
#     plt.ion()
#     plt.clf()
#     name_list = [3, 8, 9, 4, 1, 0, 5]
#     for i in name_list:#range(len(method_list)):
#         temp0 = s_list[i][:50]
#         temp1 = s_list[i][50:]
#         temp0, _ = torch.sort(temp0)
#         temp1, _ = torch.sort(temp1)
#         final_tensor = torch.cat((temp0, temp1), 0)
#         final_tensor = s_list[i][:100]
#         plt.plot(x[:100], final_tensor, label=method_list[i], linewidth=2.0, alpha = 0.7)

#     plt.xlabel('Class', fontsize=14)
#     plt.ylabel('Variance', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.axvline(x = 50, linestyle='--', color = 'r', linewidth=1.5)
#     ax = plt.gca()
#     bwith = 1.
#     ax.spines['top'].set_linewidth(bwith)
#     ax.spines['right'].set_linewidth(bwith)
#     ax.spines['bottom'].set_linewidth(bwith)
#     ax.spines['left'].set_linewidth(bwith)
#     plt.tick_params(labelsize=14)
#     plt.savefig('./Variance_Class.pdf')
#     plt.show()

# visual_variance(all_RMSSTD, method_list)

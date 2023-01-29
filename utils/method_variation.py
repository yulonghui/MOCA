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

def visual_svd(s_list):
    x = np.arange(0,512)
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    plt.clf()
    for i in range(len(s_list)):
        s_list[i] = s_list[i] / torch.sum(s_list[i])
    plt.plot(x, torch.log(s_list[3]), label=method_list[3])
    plt.plot(x, torch.log(s_list[0]), label=method_list[0])
    plt.plot(x, torch.log(s_list[1]), label=method_list[1])
    plt.plot(x, torch.log(s_list[2]), label=method_list[2])
    plt.plot(x, torch.log(s_list[4]), label=method_list[4])
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Log Singular Value', fontsize=14)
    plt.legend()

    ax = plt.gca()
    bwith = 1.
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    plt.tick_params(labelsize=14)
    plt.savefig('./svd.pdf')
    plt.show

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

def compute_variance(class_feat, m_feature):
    RMSSTD = 0
    for i_ter in range(class_feat.shape[0]):
        RMSSTD += 1 - torch.cosine_similarity(m_feature[0], class_feat[i_ter], dim=0)
    RMSSTD = RMSSTD / (class_feat.shape[0])
    return RMSSTD

EPS = 1E-20
current_task = '5'
target_type = 'target_buf_labels'
gamma_loss = '1.0'
noise_type = 'noise'
method2 = 'gaussian'
c_theta = '45.0'
para_scale = '1.0'

s_list = []
# method_list = ['drop_self', 'drop_new', 'class_mean', 'gaussian', 'none']  #'drop_self', 'drop_new', 'class_mean', 'gaussian', 
method_list = ['drop_new', 'drop_self']
for idx, method2 in enumerate(method_list):
    if method2 == 'none':
        target_type = 'new_labels'
        gamma_loss = '50.0'
        noise_type = 'adv'
    if idx == 0:
        para_scale = '1.0'
    if idx == 1:
        para_scale = '1.5'

    name = str(current_task) + '_' + target_type  + \
    '_' + str(gamma_loss) + '_' + noise_type + \
        '_' + method2 + '_' + str(c_theta) +\
        '_' + str(para_scale)

    # dirs = './out_gradient/' + 'mnist_2d/gaussian/'+ 'er_ours_3_50_task_2_model.ph'
    feat = np.load('output_7_22_all_method/buff_featurte_task_%s.npy' %name, allow_pickle=True)
    label = np.load('output_7_22_all_method/buff_labels_task_%s.npy' %name, allow_pickle=True)
    new_feat = np.load('output_7_22_all_method/new_featurte_task_%s.npy' %name, allow_pickle=True)
    new_label = np.load('output_7_22_all_method/new_labels_task_%s.npy' %name, allow_pickle=True)
    ori_feat = np.load('output_7_22_all_method/ori_buffer_feat_task_%s.npy' %name, allow_pickle=True)
    feat, label, ori_feat, new_feat, new_label = trans_feat(feat, label, ori_feat, new_feat, new_label)
    
    aug_feat_1, ori_feat_1 = copy.deepcopy(feat), copy.deepcopy(ori_feat)
    feat_noise = aug_feat_1.detach() - ori_feat_1.detach()
    
    epoch = 50
    num_per_epoch = int(feat_noise.shape[0]/epoch)
    all_feat = torch.cat((feat, new_feat), 0)
    all_label = torch.cat((label, new_label), 0)
    RMSSTD_list = []
    for i_class in range(0,100):
        mask = all_label == i_class
        class_feat = all_feat[mask] 
        
        class_feat = F.normalize(class_feat, dim=1, p=2)
        m_feature = torch.mean(class_feat, dim=0).unsqueeze(0)
        m_feature = F.normalize(m_feature, dim=1, p=2)

        RMSSTD = compute_variance(class_feat, m_feature)

        RMSSTD_list.append(RMSSTD)

    print(sum(RMSSTD_list[:80])/80, sum(RMSSTD_list[80:])/20)



# method_list = ['V-self-drop', 'V-new-drop', 'V-trans', 'V-gaussian', 'V-adv']
# visual_svd(s_list)
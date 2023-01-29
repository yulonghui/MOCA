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

EPS = 1E-20
current_task = '5'
target_type = 'target_buf_labels'
gamma_loss = '1.0'
noise_type = 'noise'
method2 = 'gaussian'
c_theta = '45.0'
para_scale = '1.0'

s_list = []
method_list = ['drop_self', 'drop_new', 'class_mean', 'gaussian', 'none']
for method2 in method_list:
    if method2 == 'none':
        target_type = 'new_labels'
        gamma_loss = '50.0'
        noise_type = 'adv'

    epoch = 50
    name = str(current_task) + '_' + target_type  + \
    '_' + str(gamma_loss) + '_' + noise_type + \
        '_' + method2 + '_' + str(c_theta) +\
        '_' + str(para_scale)
    model_name = target_type  + \
    '_' + str(gamma_loss) + '_' + noise_type + \
        '_' + method2 + '_' + str(c_theta) +\
        '_' + str(para_scale)

    model_path = './output/task_models/seq-cifar100/%s' %model_name + '/task_5_model.ph'
    net = torch.load(model_path)
    net.train()
    feat = np.load('output/buff_featurte_task_%s.npy' %name, allow_pickle=True)
    label = np.load('output/buff_labels_task_%s.npy' %name, allow_pickle=True)
    new_feat = np.load('output/new_featurte_task_%s.npy' %name, allow_pickle=True)
    new_label = np.load('output/new_labels_task_%s.npy' %name, allow_pickle=True)
    ori_feat = np.load('output/ori_buffer_feat_task_%s.npy' %name, allow_pickle=True)
    feat, label, ori_feat, new_feat, new_label = trans_feat(feat, label, ori_feat, new_feat, new_label)
    
    aug_feat_1, ori_feat_1 = copy.deepcopy(feat), copy.deepcopy(ori_feat)
    feat_noise = aug_feat_1.detach() - ori_feat_1.detach()

    feat = feat + feat_noise
    num_per_epoch = int(feat_noise.shape[0]/epoch)
    mask = label[-num_per_epoch:] <80

    feat_all = torch.cat((feat, new_feat), 0)
    label_all = torch.cat((label, new_label), 0)

    pred = net.classifier(feat_all)
    loss = F.cross_entropy(pred, label_all)
    loss.backward()

    u, s, v = torch.svd(feat.grad[-num_per_epoch:][mask])
    s_list.append(s)

method_list = ['V-self-drop', 'V-new-drop', 'V-trans', 'V-gaussian', 'V-adv']
visual_svd(s_list)
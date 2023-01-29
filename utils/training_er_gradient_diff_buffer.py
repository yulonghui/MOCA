
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
import scipy.signal as signal

def visual_svd(s_list, method_list):
    x = np.arange(0,512)
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    plt.clf()
    for i in range(len(method_list)):
    # for i in [7, 4, 3, 2, 1, 0]:
        s_list[i] = s_list[i] / torch.sum(s_list[i])
        plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.2 + 0.16*i, color = 'purple')

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
    plt.savefig('./ER_grad_svd.pdf')
    plt.show()

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

# EPS = 1E-20
# current_task = '5'
# target_type = 'new_labels'
# gamma_loss = '1.0'
# noise_type = 'noise'
# method2 = 'gaussian'
# c_theta = '45.0'
# para_scale = '1.0'
# grad_dict = {
# 'er_50'     :'buff_noise_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_200'    :'buff_noise_task_2_new_labels_1.0_noise_er_200_45.0_0.0.npy',
# 'er_2000'   :'buff_noise_task_2_new_labels_1.0_noise_er_2000_45.0_0.0.npy',
# 'er_5000'   :'buff_noise_task_2_new_labels_1.0_noise_er_5000_45.0_0.0.npy',
# 'er_10000'  :'buff_noise_task_2_new_labels_1.0_noise_er_10000_45.0_0.0.npy'
# }
# new_grad_dict = {
# 'er_50'     :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_200'    :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_200_45.0_0.0.npy',
# 'er_2000'   :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_2000_45.0_0.0.npy',
# 'er_5000'   :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_5000_45.0_0.0.npy',
# 'er_10000'  :'ori_buffer_feat_task_2_new_labels_1.0_noise_er_10000_45.0_0.0.npy'
# }

# label_dict = {
# 'er_50'     :'buff_labels_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_200'    :'buff_labels_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_2000'   :'buff_labels_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_5000'   :'buff_labels_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy',
# 'er_10000'  :'buff_labels_task_2_new_labels_1.0_noise_er_50_45.0_0.0.npy'
# }

# method_list = [
# 'er_50',
# 'er_200',
# 'er_2000',
# 'er_5000',
# 'er_10000'     
# ]

# s1 = 0
# s_list = []
# for idx, method2 in enumerate(method_list):
#     name = grad_dict[method2]
#     name = './output_er_different_buffer/' + name
#     gradient = np.load(name, allow_pickle=True)

#     name = new_grad_dict[method2]
#     name = './output_er_different_buffer/' + name
#     new_gradient = np.load(name, allow_pickle=True)

#     name = label_dict[method2]
#     name = './output_er_different_buffer/' + name
#     label = np.load(name, allow_pickle=True)
#     feat, label, ori_feat, new_feat, new_label = trans_feat(gradient, label, new_gradient, new_gradient, label)
    
#     gradient = feat
#     new_gradient = ori_feat
    
#     epoch = 50
#     num_per_epoch = int(gradient.shape[0]/epoch)
#     if idx == 0:
#         # epoch = 5
#         # num_per_epoch = int(gradient.shape[0]/epoch)
#         mask = label[-num_per_epoch:] <100 
#     else:    
#         mask = label[-num_per_epoch:] <100

#     svd_feat = torch.cat((gradient[-num_per_epoch:][mask], new_gradient[-num_per_epoch:][mask]), 0)
#     u, s1, v = torch.svd(svd_feat)
#     s_list.append(s1)

method_list = [
'ER (200)',
'ER (500)',
'ER (2000)',
'ER (5000)',
'ER (10000)',
'Joint'       
]
s_list = np.loadtxt('er_gradient_svd.txt')
s_list2 = np.zeros([7,512])
s_list3 = np.zeros([7,512])
s_list2[0] = s_list[2]
s_list2[1] = s_list[3]
s_list2[2] = s_list[1]
s_list2[3] = s_list[4]
s_list2[4] = s_list[5]
s_list2[5] = s_list[6]
s_list2[6] = s_list[0]
def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))
a = np_move_avg(s_list2[0], 3)
print(a.shape)
s_list3 = copy.deepcopy(s_list2)
# for i in range(7):
#     s_list2[i] = np_move_avg(s_list2[i], 3)
#     s_list2[i][:10] = s_list3[i][:10]
#     s_list2[i][-10:] = s_list3[i][-10:]
s_list2 = np.delete(s_list2, 1, 0)

np.savetxt('source_er_buffer_gradient.txt',s_list2)
# s_list2 = np.loadtxt('average_er_buffer_gradient.txt')

s_list2 = torch.from_numpy(s_list2)

method_list = [
'ER (200)',
'ER (500)',
'ER (2000)',
'ER (5000)',
'ER (10000)',
'Joint'       
]

visual_svd(s_list2, method_list)
# for i in range(len(s_list)):
#     s_list[i] = s_list[i].cpu().detach().numpy()
# import numpy as np
# s_list=np.array(s_list)
# np.save('er_gradient_svd.npy',s_list)


# method_list = [
# 'ER (50)',
# 'ER (200)',
# 'ER (2000)',
# 'ER (5000)',
# 'ER (10000)'    
# ]

# s_list = np.load('source_gradient_svd.npy')
# # s_list = torch.from_numpy(s_list)
# s1 = [3.0754e-01, 2.9694e-01, 2.6493e-01, 2.5012e-01, 2.3098e-01, 2.1519e-01,
#         1.9815e-01, 1.9279e-01, 1.7776e-01, 1.6697e-01, 1.5623e-01, 1.5035e-01,
#         1.4430e-01, 1.3601e-01, 1.2457e-01, 1.1740e-01, 1.1490e-01, 1.0365e-01,
#         9.7824e-02, 9.0633e-02, 8.7719e-02, 8.5711e-02, 8.1260e-02, 7.9664e-02,
#         7.4037e-02, 7.2626e-02, 6.8902e-02, 6.4957e-02, 6.4312e-02, 5.9881e-02,
#         5.6572e-02, 5.1805e-02, 5.0400e-02, 4.9481e-02, 4.7427e-02, 4.6315e-02,
#         4.5118e-02, 4.4918e-02, 4.2412e-02, 3.8979e-02, 3.7640e-02, 3.4072e-02,
#         3.3329e-02, 3.1242e-02, 2.9187e-02, 2.7545e-02, 2.4852e-02, 2.4226e-02,
#         2.1984e-02, 2.0110e-02, 1.7838e-02, 1.6983e-02, 1.4438e-02, 1.4226e-02,
#         1.2885e-02, 1.2450e-02, 1.1688e-02, 1.0773e-02, 1.0629e-02, 1.0178e-02,
#         8.9535e-03, 8.2712e-03, 7.5165e-03, 6.3186e-03, 5.8792e-03, 5.4375e-03,
#         5.1686e-03, 4.9887e-03, 4.6630e-03, 4.4596e-03, 3.9627e-03, 3.8842e-03,
#         3.7746e-03, 3.5330e-03, 3.5165e-03, 3.4235e-03, 3.2218e-03, 3.1227e-03,
#         2.7839e-03, 2.6367e-03, 2.6102e-03, 2.5538e-03, 2.4914e-03, 2.3850e-03,
#         2.3492e-03, 2.2371e-03, 2.1641e-03, 2.1091e-03, 2.0403e-03, 1.9363e-03,
#         1.9054e-03, 1.8272e-03, 1.7657e-03, 1.7477e-03, 1.6799e-03, 1.6373e-03,
#         1.6195e-03, 1.5995e-03, 1.5885e-03, 1.5489e-03, 1.5358e-03, 1.5155e-03,
#         1.4516e-03, 1.4098e-03, 1.3950e-03, 1.3583e-03, 1.3520e-03, 1.3196e-03,
#         1.3136e-03, 1.2947e-03, 1.2725e-03, 1.2628e-03, 1.2365e-03, 1.2100e-03,
#         1.1722e-03, 1.1677e-03, 1.1466e-03, 1.1412e-03, 1.1312e-03, 1.1046e-03,
#         1.0944e-03, 1.0841e-03, 1.0784e-03, 1.0550e-03, 1.0439e-03, 1.0258e-03,
#         1.0192e-03, 1.0147e-03, 9.8441e-04, 9.6823e-04, 9.6215e-04, 9.5704e-04,
#         9.4087e-04, 9.2731e-04, 9.0717e-04, 9.0265e-04, 8.9121e-04, 8.8809e-04,
#         8.8073e-04, 8.7305e-04, 8.6129e-04, 8.5194e-04, 8.4965e-04, 8.2954e-04,
#         8.2452e-04, 8.2124e-04, 8.1497e-04, 8.0787e-04, 7.9889e-04, 7.9673e-04,
#         7.7926e-04, 7.7399e-04, 7.7171e-04, 7.6248e-04, 7.5137e-04, 7.4165e-04,
#         7.3479e-04, 7.2974e-04, 7.2417e-04, 7.1646e-04, 7.1040e-04, 7.0569e-04,
#         6.9761e-04, 6.9479e-04, 6.8405e-04, 6.7843e-04, 6.6701e-04, 6.6498e-04,
#         6.5950e-04, 6.5548e-04, 6.4947e-04, 6.4457e-04, 6.4136e-04, 6.3635e-04,
#         6.3251e-04, 6.2844e-04, 6.2276e-04, 6.1911e-04, 6.1188e-04, 6.0941e-04,
#         6.0446e-04, 5.9993e-04, 5.9526e-04, 5.9215e-04, 5.8466e-04, 5.8284e-04,
#         5.7853e-04, 5.7279e-04, 5.6974e-04, 5.6491e-04, 5.6227e-04, 5.5410e-04,
#         5.5049e-04, 5.4599e-04, 5.4299e-04, 5.4177e-04, 5.3722e-04, 5.3288e-04,
#         5.2724e-04, 5.2683e-04, 5.2196e-04, 5.1896e-04, 5.1466e-04, 5.1098e-04,
#         5.0720e-04, 5.0583e-04, 4.9920e-04, 4.9705e-04, 4.9389e-04, 4.9181e-04,
#         4.8856e-04, 4.8219e-04, 4.7933e-04, 4.7824e-04, 4.7589e-04, 4.7094e-04,
#         4.6786e-04, 4.6133e-04, 4.5946e-04, 4.5899e-04, 4.5115e-04, 4.4602e-04,
#         4.4436e-04, 4.4187e-04, 4.3934e-04, 4.3752e-04, 4.3575e-04, 4.3418e-04,
#         4.2970e-04, 4.2447e-04, 4.2364e-04, 4.2285e-04, 4.1897e-04, 4.1801e-04,
#         4.1389e-04, 4.1081e-04, 4.0608e-04, 4.0456e-04, 4.0262e-04, 4.0026e-04,
#         3.9678e-04, 3.9540e-04, 3.9026e-04, 3.8928e-04, 3.8662e-04, 3.8463e-04,
#         3.8369e-04, 3.8150e-04, 3.7907e-04, 3.7358e-04, 3.7276e-04, 3.7175e-04,
#         3.7073e-04, 3.6647e-04, 3.6501e-04, 3.6428e-04, 3.6218e-04, 3.6086e-04,
#         3.5815e-04, 3.5545e-04, 3.5201e-04, 3.5079e-04, 3.4669e-04, 3.4465e-04,
#         3.4276e-04, 3.4038e-04, 3.3970e-04, 3.3766e-04, 3.3673e-04, 3.3473e-04,
#         3.3208e-04, 3.3058e-04, 3.2970e-04, 3.2714e-04, 3.2615e-04, 3.2322e-04,
#         3.2146e-04, 3.2035e-04, 3.1573e-04, 3.1529e-04, 3.1425e-04, 3.1233e-04,
#         3.1051e-04, 3.0821e-04, 3.0740e-04, 3.0588e-04, 3.0376e-04, 2.9943e-04,
#         2.9900e-04, 2.9751e-04, 2.9662e-04, 2.9407e-04, 2.9296e-04, 2.9075e-04,
#         2.8932e-04, 2.8875e-04, 2.8753e-04, 2.8404e-04, 2.8265e-04, 2.8189e-04,
#         2.8115e-04, 2.7850e-04, 2.7642e-04, 2.7556e-04, 2.7369e-04, 2.6909e-04,
#         2.6865e-04, 2.6794e-04, 2.6649e-04, 2.6443e-04, 2.6365e-04, 2.6177e-04,
#         2.6037e-04, 2.5932e-04, 2.5810e-04, 2.5733e-04, 2.5577e-04, 2.5525e-04,
#         2.5243e-04, 2.5175e-04, 2.5068e-04, 2.4928e-04, 2.4720e-04, 2.4632e-04,
#         2.4457e-04, 2.4363e-04, 2.4254e-04, 2.4199e-04, 2.3907e-04, 2.3888e-04,
#         2.3738e-04, 2.3685e-04, 2.3565e-04, 2.3428e-04, 2.3285e-04, 2.3107e-04,
#         2.3031e-04, 2.2892e-04, 2.2659e-04, 2.2550e-04, 2.2439e-04, 2.2306e-04,
#         2.2248e-04, 2.2037e-04, 2.1951e-04, 2.1779e-04, 2.1732e-04, 2.1567e-04,
#         2.1412e-04, 2.1260e-04, 2.1064e-04, 2.1046e-04, 2.0955e-04, 2.0770e-04,
#         2.0686e-04, 2.0632e-04, 2.0559e-04, 2.0390e-04, 2.0273e-04, 2.0159e-04,
#         2.0020e-04, 1.9896e-04, 1.9866e-04, 1.9826e-04, 1.9722e-04, 1.9603e-04,
#         1.9567e-04, 1.9472e-04, 1.9305e-04, 1.9221e-04, 1.9163e-04, 1.9053e-04,
#         1.8975e-04, 1.8916e-04, 1.8844e-04, 1.8657e-04, 1.8503e-04, 1.8442e-04,
#         1.8398e-04, 1.8237e-04, 1.8187e-04, 1.7927e-04, 1.7893e-04, 1.7801e-04,
#         1.7691e-04, 1.7649e-04, 1.7595e-04, 1.7518e-04, 1.7390e-04, 1.7273e-04,
#         1.7236e-04, 1.7206e-04, 1.7052e-04, 1.6974e-04, 1.6877e-04, 1.6758e-04,
#         1.6633e-04, 1.6584e-04, 1.6443e-04, 1.6403e-04, 1.6237e-04, 1.6194e-04,
#         1.6139e-04, 1.6067e-04, 1.5875e-04, 1.5783e-04, 1.5779e-04, 1.5631e-04,
#         1.5501e-04, 1.5448e-04, 1.5344e-04, 1.5290e-04, 1.5268e-04, 1.5125e-04,
#         1.5086e-04, 1.5009e-04, 1.4997e-04, 1.4847e-04, 1.4770e-04, 1.4600e-04,
#         1.4580e-04, 1.4527e-04, 1.4432e-04, 1.4400e-04, 1.4273e-04, 1.4143e-04,
#         1.4090e-04, 1.4035e-04, 1.3949e-04, 1.3833e-04, 1.3681e-04, 1.3628e-04,
#         1.3583e-04, 1.3500e-04, 1.3463e-04, 1.3339e-04, 1.3281e-04, 1.3219e-04,
#         1.3116e-04, 1.3036e-04, 1.3010e-04, 1.2959e-04, 1.2843e-04, 1.2731e-04,
#         1.2629e-04, 1.2582e-04, 1.2556e-04, 1.2508e-04, 1.2450e-04, 1.2379e-04,
#         1.2278e-04, 1.2198e-04, 1.2182e-04, 1.2153e-04, 1.2051e-04, 1.1985e-04,
#         1.1937e-04, 1.1840e-04, 1.1734e-04, 1.1685e-04, 1.1604e-04, 1.1486e-04,
#         1.1432e-04, 1.1386e-04, 1.1330e-04, 1.1230e-04, 1.1194e-04, 1.1082e-04,
#         1.1003e-04, 1.0916e-04, 1.0883e-04, 1.0801e-04, 1.0737e-04, 1.0705e-04,
#         1.0585e-04, 1.0540e-04, 1.0379e-04, 1.0324e-04, 1.0284e-04, 1.0164e-04,
#         1.0113e-04, 1.0045e-04, 9.9174e-05, 9.8610e-05, 9.8060e-05, 9.7274e-05,
#         9.6894e-05, 9.6483e-05, 9.6172e-05, 9.4769e-05, 9.4105e-05, 9.3731e-05,
#         9.3032e-05, 9.1586e-05, 9.1304e-05, 9.0827e-05, 9.0376e-05, 8.9249e-05,
#         8.8516e-05, 8.7869e-05, 8.6626e-05, 8.6142e-05, 8.5672e-05, 8.3668e-05,
#         8.3310e-05, 8.2817e-05, 8.1712e-05, 8.0680e-05, 7.9070e-05, 7.7129e-05,
#         7.4946e-05, 7.4809e-05]
# s_list[0] = np.array(s1)
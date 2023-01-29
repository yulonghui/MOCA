
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
    for i in [5, 3, 6, 7, 1, 2, 0, 4]:
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
        if method_list[i] == 'V-vmf':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9, color='chocolate')
            continue
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
    plt.savefig('./vmf_svd.pdf')
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

EPS = 1E-20
current_task = '5'
target_type = 'target_buf_labels'
gamma_loss = '1.0'
noise_type = 'noise'
method2 = 'gaussian'
c_theta = '45.0'
para_scale = '1.0'
feat_dict = {
'adv'           :'buff_noise_task_2_new_labels_50.0_adv_none_45.0_1.0.npy',
'class_mean'    :'buff_noise_task_2_target_buf_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'buff_noise_task_2_target_buf_labels_1.0_noise_drop_new_45.0_1.0.npy',
'drop_self'     :'buff_noise_task_2_target_buf_labels_1.0_noise_drop_self_45.0_1.0.npy',
'er_30000'      :'buff_noise_task_2_target_buf_labels_1.0_noise_er_30000_45.0_0.0.npy',
# 'er_2000'        :'buff_noise_task_2_target_buf_labels_1.0_noise_er_2000_45.0_0.0.npy',
'er_200'        :'buff_noise_task_2_target_buf_labels_1.0_noise_er_200_45.0_0.0.npy',
'gaussian'      :'buff_noise_task_2_target_buf_labels_1.0_noise_gaussian_45.0_1.0.npy',
'vmf'           :'buff_noise_task_2_new_labels_100.0_noise_vmf_200_100_45.0_1.0.npy',
}

new_feat_dict = {
'adv'           :'ori_buffer_feat_task_2_new_labels_50.0_adv_none_45.0_1.0.npy',
'class_mean'    :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_drop_new_45.0_1.0.npy',
'drop_self'     :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_drop_self_45.0_1.0.npy',
'er_30000'      :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_er_30000_45.0_0.0.npy',
'er_200'        :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_er_200_45.0_0.0.npy',
'gaussian'      :'ori_buffer_feat_task_2_target_buf_labels_1.0_noise_gaussian_45.0_1.0.npy',
'vmf'           :'ori_buffer_feat_task_2_new_labels_100.0_noise_vmf_200_100_45.0_1.0.npy',
}

label_dict = {
'adv'           :'buff_labels_task_2_new_labels_50.0_adv_none_45.0_1.0.npy',
'class_mean'    :'buff_labels_task_2_target_buf_labels_1.0_noise_class_mean_45.0_1.0.npy',
'drop_new'      :'buff_labels_task_2_target_buf_labels_1.0_noise_drop_new_45.0_1.0.npy',
'drop_self'     :'buff_labels_task_2_target_buf_labels_1.0_noise_drop_self_45.0_1.0.npy',
'er_30000'      :'buff_labels_task_2_target_buf_labels_1.0_noise_er_30000_45.0_0.0.npy',
# 'er_2000'        :'buff_labels_task_2_target_buf_labels_1.0_noise_er_2000_45.0_0.0.npy',
'er_200'        :'buff_labels_task_2_target_buf_labels_1.0_noise_er_200_45.0_0.0.npy',
'gaussian'      :'buff_labels_task_2_target_buf_labels_1.0_noise_gaussian_45.0_1.0.npy',
'vmf'           :'buff_labels_task_2_new_labels_100.0_noise_vmf_200_100_45.0_1.0.npy',
}
s_list = []
method_list = ['adv',       
'class_mean',
'drop_new'  ,
'drop_self' ,
'er_30000'  ,
# 'er_2000'   ,
'er_200'    ,
'gaussian'  ]

# method_list = ['vmf']
# s1 = 0
# for idx, method2 in enumerate(method_list):
#     if method2 == 'adv':
#         method2 = 'class_mean'
#     name = feat_dict[method2]
#     name = './output_vmf/' + name
#     gradient = np.load(name, allow_pickle=True)

#     name = new_feat_dict[method2]
#     name = './output_vmf/' + name
#     new_gradient = np.load(name, allow_pickle=True)

#     name = label_dict[method2]
#     name = './output_vmf/' + name
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


print(s_list)
s_vmf = torch.tensor([1.4397e+00, 5.3825e-01, 5.1877e-01, 5.1528e-01, 4.5640e-01, 4.2078e-01,
        3.8525e-01, 3.7440e-01, 3.6255e-01, 3.4863e-01, 3.4152e-01, 3.2087e-01,
        3.1700e-01, 3.0190e-01, 2.8858e-01, 2.7711e-01, 2.6851e-01, 2.5351e-01,
        2.4586e-01, 2.4018e-01, 2.3482e-01, 2.3174e-01, 2.1976e-01, 2.1508e-01,
        2.1105e-01, 2.0325e-01, 1.9170e-01, 1.8823e-01, 1.8061e-01, 1.7452e-01,
        1.6941e-01, 1.6760e-01, 1.6473e-01, 1.5904e-01, 1.5739e-01, 1.4289e-01,
        1.4095e-01, 1.3458e-01, 1.2377e-01, 1.2029e-01, 1.1364e-01, 1.1143e-01,
        1.0435e-01, 1.0013e-01, 9.5467e-02, 9.3088e-02, 9.0240e-02, 8.6443e-02,
        8.4506e-02, 7.4152e-02, 6.8657e-02, 6.3023e-02, 5.1785e-02, 4.9750e-02,
        4.7472e-02, 4.2688e-02, 4.0837e-02, 3.9865e-02, 3.6561e-02, 3.5457e-02,
        3.3269e-02, 3.0372e-02, 2.8867e-02, 2.5401e-02, 2.5076e-02, 2.4713e-02,
        2.3551e-02, 2.2478e-02, 2.1064e-02, 2.0696e-02, 1.9094e-02, 1.8812e-02,
        1.8537e-02, 1.7633e-02, 1.6620e-02, 1.6384e-02, 1.5529e-02, 1.5335e-02,
        1.4163e-02, 1.3810e-02, 1.3628e-02, 1.3303e-02, 1.3147e-02, 1.2254e-02,
        1.2028e-02, 1.1632e-02, 1.1415e-02, 1.0932e-02, 1.0600e-02, 1.0326e-02,
        9.8781e-03, 9.7714e-03, 9.4502e-03, 9.2571e-03, 8.7732e-03, 8.2669e-03,
        8.0024e-03, 6.6300e-03, 6.4517e-03, 5.6784e-03, 5.1826e-03, 5.0348e-03,
        4.6783e-03, 4.5159e-03, 4.4488e-03, 4.3901e-03, 4.2578e-03, 4.1620e-03,
        3.9866e-03, 3.9258e-03, 3.7289e-03, 3.6555e-03, 3.5382e-03, 3.4716e-03,
        3.4448e-03, 3.4164e-03, 3.3435e-03, 3.2802e-03, 3.2451e-03, 3.1640e-03,
        3.1508e-03, 3.1472e-03, 3.1197e-03, 3.0626e-03, 3.0349e-03, 2.9992e-03,
        2.9650e-03, 2.9315e-03, 2.9112e-03, 2.8605e-03, 2.8316e-03, 2.8149e-03,
        2.7705e-03, 2.7602e-03, 2.7393e-03, 2.7118e-03, 2.6959e-03, 2.6661e-03,
        2.6539e-03, 2.6152e-03, 2.5988e-03, 2.5529e-03, 2.5483e-03, 2.5407e-03,
        2.5086e-03, 2.4894e-03, 2.4841e-03, 2.4505e-03, 2.4494e-03, 2.4091e-03,
        2.4029e-03, 2.3914e-03, 2.3671e-03, 2.3458e-03, 2.3325e-03, 2.3112e-03,
        2.3059e-03, 2.2803e-03, 2.2626e-03, 2.2540e-03, 2.2350e-03, 2.2300e-03,
        2.2139e-03, 2.1831e-03, 2.1704e-03, 2.1688e-03, 2.1535e-03, 2.1435e-03,
        2.1359e-03, 2.1254e-03, 2.1170e-03, 2.1087e-03, 2.0975e-03, 2.0910e-03,
        2.0747e-03, 2.0666e-03, 2.0576e-03, 2.0482e-03, 2.0334e-03, 2.0204e-03,
        2.0009e-03, 1.9956e-03, 1.9796e-03, 1.9746e-03, 1.9622e-03, 1.9548e-03,
        1.9471e-03, 1.9405e-03, 1.9320e-03, 1.9251e-03, 1.9150e-03, 1.9103e-03,
        1.8993e-03, 1.8956e-03, 1.8772e-03, 1.8667e-03, 1.8626e-03, 1.8620e-03,
        1.8546e-03, 1.8497e-03, 1.8347e-03, 1.8308e-03, 1.8252e-03, 1.8143e-03,
        1.8078e-03, 1.7987e-03, 1.7879e-03, 1.7777e-03, 1.7675e-03, 1.7590e-03,
        1.7562e-03, 1.7491e-03, 1.7390e-03, 1.7306e-03, 1.7257e-03, 1.7143e-03,
        1.7114e-03, 1.7051e-03, 1.6943e-03, 1.6873e-03, 1.6860e-03, 1.6832e-03,
        1.6730e-03, 1.6682e-03, 1.6602e-03, 1.6556e-03, 1.6508e-03, 1.6399e-03,
        1.6368e-03, 1.6283e-03, 1.6250e-03, 1.6223e-03, 1.6088e-03, 1.6031e-03,
        1.5987e-03, 1.5902e-03, 1.5878e-03, 1.5858e-03, 1.5768e-03, 1.5676e-03,
        1.5604e-03, 1.5575e-03, 1.5539e-03, 1.5481e-03, 1.5432e-03, 1.5392e-03,
        1.5318e-03, 1.5307e-03, 1.5248e-03, 1.5147e-03, 1.5142e-03, 1.5052e-03,
        1.5004e-03, 1.4975e-03, 1.4902e-03, 1.4870e-03, 1.4835e-03, 1.4802e-03,
        1.4750e-03, 1.4715e-03, 1.4614e-03, 1.4569e-03, 1.4527e-03, 1.4512e-03,
        1.4435e-03, 1.4381e-03, 1.4372e-03, 1.4282e-03, 1.4233e-03, 1.4167e-03,
        1.4137e-03, 1.4100e-03, 1.4055e-03, 1.4033e-03, 1.3993e-03, 1.3966e-03,
        1.3918e-03, 1.3864e-03, 1.3853e-03, 1.3804e-03, 1.3767e-03, 1.3710e-03,
        1.3681e-03, 1.3667e-03, 1.3587e-03, 1.3536e-03, 1.3504e-03, 1.3486e-03,
        1.3434e-03, 1.3415e-03, 1.3322e-03, 1.3270e-03, 1.3240e-03, 1.3196e-03,
        1.3176e-03, 1.3123e-03, 1.3106e-03, 1.3070e-03, 1.3040e-03, 1.3013e-03,
        1.2934e-03, 1.2898e-03, 1.2879e-03, 1.2844e-03, 1.2825e-03, 1.2793e-03,
        1.2759e-03, 1.2722e-03, 1.2705e-03, 1.2625e-03, 1.2604e-03, 1.2550e-03,
        1.2536e-03, 1.2455e-03, 1.2438e-03, 1.2429e-03, 1.2405e-03, 1.2380e-03,
        1.2329e-03, 1.2297e-03, 1.2270e-03, 1.2224e-03, 1.2223e-03, 1.2159e-03,
        1.2140e-03, 1.2129e-03, 1.2116e-03, 1.2097e-03, 1.1997e-03, 1.1945e-03,
        1.1943e-03, 1.1926e-03, 1.1897e-03, 1.1843e-03, 1.1835e-03, 1.1794e-03,
        1.1779e-03, 1.1752e-03, 1.1690e-03, 1.1668e-03, 1.1657e-03, 1.1594e-03,
        1.1570e-03, 1.1560e-03, 1.1478e-03, 1.1465e-03, 1.1428e-03, 1.1412e-03,
        1.1375e-03, 1.1330e-03, 1.1324e-03, 1.1317e-03, 1.1237e-03, 1.1202e-03,
        1.1174e-03, 1.1149e-03, 1.1141e-03, 1.1091e-03, 1.1049e-03, 1.1026e-03,
        1.1003e-03, 1.0990e-03, 1.0944e-03, 1.0899e-03, 1.0890e-03, 1.0875e-03,
        1.0813e-03, 1.0802e-03, 1.0773e-03, 1.0731e-03, 1.0723e-03, 1.0677e-03,
        1.0672e-03, 1.0661e-03, 1.0640e-03, 1.0620e-03, 1.0563e-03, 1.0558e-03,
        1.0532e-03, 1.0514e-03, 1.0463e-03, 1.0445e-03, 1.0428e-03, 1.0414e-03,
        1.0375e-03, 1.0353e-03, 1.0349e-03, 1.0294e-03, 1.0272e-03, 1.0243e-03,
        1.0203e-03, 1.0183e-03, 1.0148e-03, 1.0120e-03, 1.0080e-03, 1.0066e-03,
        1.0048e-03, 1.0030e-03, 1.0016e-03, 9.9854e-04, 9.9488e-04, 9.9388e-04,
        9.8982e-04, 9.8927e-04, 9.8611e-04, 9.8111e-04, 9.7958e-04, 9.7694e-04,
        9.7495e-04, 9.7304e-04, 9.7047e-04, 9.6832e-04, 9.6398e-04, 9.6267e-04,
        9.6042e-04, 9.5792e-04, 9.5593e-04, 9.5101e-04, 9.4992e-04, 9.4817e-04,
        9.4746e-04, 9.4341e-04, 9.4195e-04, 9.3863e-04, 9.3766e-04, 9.3425e-04,
        9.3166e-04, 9.2996e-04, 9.2771e-04, 9.2501e-04, 9.2436e-04, 9.1922e-04,
        9.1790e-04, 9.1742e-04, 9.1291e-04, 9.0851e-04, 9.0638e-04, 9.0317e-04,
        9.0223e-04, 8.9973e-04, 8.9823e-04, 8.9616e-04, 8.9314e-04, 8.9075e-04,
        8.8913e-04, 8.8847e-04, 8.8474e-04, 8.8093e-04, 8.7827e-04, 8.7638e-04,
        8.7570e-04, 8.7387e-04, 8.7266e-04, 8.7035e-04, 8.6651e-04, 8.6294e-04,
        8.6240e-04, 8.5876e-04, 8.5774e-04, 8.5664e-04, 8.5137e-04, 8.4970e-04,
        8.4517e-04, 8.4362e-04, 8.4186e-04, 8.3857e-04, 8.3711e-04, 8.3506e-04,
        8.3147e-04, 8.3040e-04, 8.2921e-04, 8.2674e-04, 8.2403e-04, 8.2279e-04,
        8.1890e-04, 8.1750e-04, 8.1332e-04, 8.0920e-04, 8.0839e-04, 8.0649e-04,
        8.0461e-04, 8.0120e-04, 8.0029e-04, 7.9985e-04, 7.9506e-04, 7.9364e-04,
        7.9082e-04, 7.8662e-04, 7.8396e-04, 7.8269e-04, 7.8173e-04, 7.7858e-04,
        7.7216e-04, 7.7166e-04, 7.6841e-04, 7.6717e-04, 7.6532e-04, 7.6090e-04,
        7.5810e-04, 7.5630e-04, 7.5422e-04, 7.5102e-04, 7.4577e-04, 7.4448e-04,
        7.3977e-04, 7.3754e-04, 7.3267e-04, 7.3233e-04, 7.2456e-04, 7.1868e-04,
        7.1514e-04, 7.0261e-04]).unsqueeze(0)

print(s_vmf.shape)
s_list2 = np.loadtxt('perturb_gradient.txt.txt')

s_list2 = torch.from_numpy(s_list2)
s_list2 = torch.cat((s_list2, s_vmf), 0)
print(s_list2.shape)
method_list = [
'V-adv',       
'V-trans',
'V-new-drop'  ,
'V-self-drop' ,
'Joint'  ,
'ER'    ,
'V-gaussian'  ,
'V-vmf' ]
s_list2 = s_list2.cpu().detach().numpy()
np.savetxt('all_perturb_gradient_31.txt',s_list2)
# visual_svd(s_list2, method_list)

# s_list = np.load('method_svd.npy')
# s_list = torch.from_numpy(s_list)
# visual_svd(s_list, method_list)
# # for i in range(len(s_list)):
# #     s_list[i] = s_list[i].cpu().detach().numpy()
# # import numpy as np
# # s_list=np.array(s_list)
# # np.save('t_svd.npy',s_list)
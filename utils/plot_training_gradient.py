
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
        if method_list[i] == 'WAP':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9, color = 'purple')
            continue
        if method_list[i] == 'DOA-new':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.65, color='purple')
            continue
        if method_list[i] == 'VT':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.4, color='purple')
            continue
        if method_list[i] == 'vMF':
            plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9, color='chocolate')
            continue
        plt.plot(x[:100], torch.log(s_list[i])[:100], label=method_list[i], linewidth=3.0, alpha = 0.9)

    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Log Singular Value', fontsize=14)
    plt.legend(fontsize=12, edgecolor='black')
    # plt.legend.get_frame().set_edgecolor('black')
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


s_list2 = np.loadtxt('all_perturb_gradient_31.txt')

s_list2 = torch.from_numpy(s_list2)

method_list = [
'WAP',       
'VT',
'DOA-new'  ,
'DOA-old' ,
'Joint'  ,
'ER'    ,
'Gaussian'  ,
'vMF' ]

# s_list2 = s_list2.cpu().detach().numpy()
# np.savetxt('all_perturb_gradient_31.txt',s_list2)
visual_svd(s_list2, method_list)
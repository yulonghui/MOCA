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



data = [29.33, 34.08, 38.76, 41.02]
labels = ['Original \n Manifold Mixup', 'Adapted \n Manifold Mixup', 'model-agnostic \n MOCA', 'model-based \n MOCA']
plt.figure(figsize=(8,5))
plt.bar(range(len(data)), data, tick_label=labels, width = 0.35, alpha = 0.8, edgecolor='black')
plt.ylim(30, 42)
plt.axhline(y = 31.08, linestyle='--', color = 'r', linewidth=1.5)
plt.tick_params(labelsize=12)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Method', fontsize=14)
ax = plt.gca()
bwith = 1.
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.yaxis.set_ticks([26, 32, 38, 44])
plt.show()
plt.savefig('./mixup_acc.pdf', bbox_inches = 'tight')



def visual_svd(s_list, method_list):
    x = np.arange(0,512)
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    plt.clf()
    for i in range(len(s_list)):
        if 'V-vmf' in method_list[i]:
            continue
        s_list[i] = s_list[i] / torch.sum(s_list[i])
        plt.plot(x, torch.log(s_list[i]), label=method_list[i])
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
    plt.savefig('./aug_feat.pdf')
    plt.show

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



data = [33.67, 37.29, 38.76, 38.75, 39.78, 41.02]
labels = ['DOA-self', 'Gaussian', 'vMF', 'DOA', 'VT', 'WAP']
plt.figure(figsize=(8,4))
plt.figure(figsize=(8,4))
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
ax.yaxis.set_ticks([30, 34, 38, 42])
plt.show()
plt.savefig('./all_method_acc.pdf', bbox_inches = 'tight')




from matplotlib.ticker import MaxNLocator
data = [33.54, 51.279, 68.75, 41.17, 35.72, 37.08]
labels = ['DOA-self', 'Gaussian', 'vMF', 'DOA', 'VT', 'WAP']
plt.figure(figsize=(8,4))
plt.bar(range(len(data)), data, tick_label=labels, width = 0.3, alpha = 0.8, edgecolor='black')
plt.ylim(29, 46)
plt.axhline(y = 30.12, linestyle='--', color = 'r', linewidth=1.5)
plt.tick_params(labelsize=12)
plt.ylabel('Old Intra-class Variance', fontsize=14)
plt.xlabel('Method', fontsize=14)
ax = plt.gca()
bwith = 1.
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.yaxis.set_ticks([30, 34, 38, 42, 46]) 
plt.show()
plt.savefig('./all_method_variance_exp.pdf', bbox_inches = 'tight')

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

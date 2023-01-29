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


Model_agnostic = [30.48, 31.67, 32.58, 34.8, 37.98, 35.59]
Model_based = [34.54, 38.0, 40.55, 39.94, 38.55, 20]
# plt.figure(figsize=(8,8))
# title = 'CIFAR-100'
# plt.title(r'Perturbation Magnitude $\lambda$')
# plt.title(title, fontsize=14)
plt.ylim(30, 45)
plt.xlabel(r'Perturbation Angular', fontsize=14)
plt.ylabel("Accuracy", fontsize=14)

plt.grid(linewidth = 1.5)
x = ['5', '15', '30', '45', '60', '75']
plt.plot(x, Model_agnostic, alpha=0.8, label="Model-agnostic MOCA", linewidth=3, marker= 'o')
plt.plot(x, Model_based, alpha=0.8, label="Model-based MOCA", linewidth=3, marker= 'o')
plt.tick_params(labelsize=14)
plt.axhline(y = 31.12, linestyle='--', color = 'r', linewidth=1.5)
ax = plt.gca()
bwith = 1.
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
plt.legend(loc="best", fontsize=14, edgecolor='black')
plt.show()
plt.savefig('./acc_perturbed_angular.pdf', bbox_inches = 'tight')

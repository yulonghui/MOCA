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


# Baseline = [83.05, 62.15, 46.12, 40.19, 31.08]
# Self_drop = [83.05, 61.32, 49.02, 41.62, 33.67]
# gaussian = [83.05, 64.05, 51.13, 42.66, 37.29]
# vMF = [83.05, 66.45, 53.22, 44.17, 38.76]
# new_drop = [83.05, 65.15, 52.13, 43.93, 38.75]
# Trans = [83.05, 65.88, 53.10, 45.30, 39.78]
# Adv = [83.05, 66.92, 54.58, 46.72, 41.02]
# # plt.figure(figsize=(8,8))
# title = 'CIFAR-100'
# plt.title(title, fontsize=14)
# plt.ylim(30, 68)
# plt.xlabel("Continual Task", fontsize=14)
# plt.ylabel("Accuracy", fontsize=14)

# plt.grid(linewidth = 1.5)
# x = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5']
# plt.plot(x, Baseline, alpha=0.8, label="ER", linewidth=3, marker= 'o')
# plt.plot(x, Self_drop, alpha=0.8, label="DOA-self", linewidth=3, marker= 'o')
# plt.plot(x, gaussian, alpha=0.8, label="Gaussian", linewidth=3, marker= 'o')
# plt.plot(x, vMF, alpha=0.8, label="vMF", linewidth=3, marker= 'o')
# plt.plot(x, new_drop, alpha=0.8, label="DOA", linewidth=3, marker= 'o')
# plt.plot(x, Trans, alpha=0.8, label="VT", linewidth=3, marker= 'o')
# plt.plot(x, Adv, alpha=0.8, label="WAP", linewidth=3, marker= 'o')
# plt.tick_params(labelsize=14)
# ax = plt.gca()
# bwith = 1.
# ax.spines['top'].set_linewidth(bwith)
# ax.spines['right'].set_linewidth(bwith)
# ax.spines['bottom'].set_linewidth(bwith)
# ax.spines['left'].set_linewidth(bwith)
# plt.legend(loc="best", fontsize=14, edgecolor='black')
# plt.show()
# plt.savefig('./acc_curve_%s.pdf' %title, bbox_inches = 'tight')



# Baseline = [74.6, 52.2, 40.63, 32.22, 24.6, 20.38, 16.71, 14.68, 12.17, 11.24]
# Self_drop = [74.6, 53.75, 40.33, 32.42, 26.94, 21.9, 17.17, 15.41, 12.31, 11.15]
# gaussian = [74.6, 59.1, 46.63, 37.52, 28.74, 25.17, 20.56, 19.12, 16.21, 14.01]
# vMF = [74.6, 57.21, 43.24, 34.55, 29.62, 24.27, 19.24, 19.95, 16.87, 14.62]
# new_drop = [74.6, 55.85, 42.53, 34.8, 25.06, 21.8, 18.9, 19.06, 15.58, 14.96]
# Trans = [74.6, 56.0, 44.63, 36.58, 31.1, 23.67, 19.93, 18.51, 14.91, 15.03]
# Adv = [74.6, 59.2, 45.77, 35.58, 30.12, 26.52, 23.93, 20.35, 18.05, 16.68]
# # plt.figure(figsize=(8,8))
# title = 'TinyImageNet'
# plt.title(title, fontsize=14)
# plt.ylim(0, 60)
# plt.xlabel("Continual Task", fontsize=14)
# plt.ylabel("Accuracy", fontsize=14)

# plt.grid(linewidth = 1.5)
# x = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8', 'Task9', 'Task10']
# plt.plot(x, Baseline, alpha=0.8, label="ER", linewidth=3, marker= 'o')
# plt.plot(x, Self_drop, alpha=0.8, label="DOA-self", linewidth=3, marker= 'o')
# plt.plot(x, gaussian, alpha=0.8, label="Gaussian", linewidth=3, marker= 'o')
# plt.plot(x, vMF, alpha=0.8, label="vMF", linewidth=3, marker= 'o')
# plt.plot(x, new_drop, alpha=0.8, label="DOA", linewidth=3, marker= 'o')
# plt.plot(x, Trans, alpha=0.8, label="VT", linewidth=3, marker= 'o')
# plt.plot(x, Adv, alpha=0.8, label="WAP", linewidth=3, marker= 'o')
# plt.tick_params(labelsize=10)
# ax = plt.gca()
# bwith = 1.
# ax.spines['top'].set_linewidth(bwith)
# ax.spines['right'].set_linewidth(bwith)
# ax.spines['bottom'].set_linewidth(bwith)
# ax.spines['left'].set_linewidth(bwith)
# plt.legend(loc="best", fontsize=14, edgecolor='black')
# plt.show()
# plt.savefig('./acc_curve_%s.pdf' %title, bbox_inches = 'tight')



Baseline = [98.25, 90.22, 74.22, 71.03, 62.53]
Self_drop = [98.25, 90.7, 79.23, 70.88, 72.18]
gaussian = [98.25, 88.82, 78.68, 74.23, 67.85]
vMF = [98.25, 88.82, 78.87, 76.18, 71.58]
new_drop = [98.25, 90.15, 72.68, 75.42, 67.88]
Trans = [98.25, 89.5, 78.98, 69.24, 71.66]
Adv = [98.25, 91.35, 80.03, 74.24, 72.99]

# plt.figure(figsize=(8,8))
title = 'CIFAR-10'
plt.title(title, fontsize=14)
plt.ylim(50, 100)
plt.xlabel("Continual Task", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)

plt.grid(linewidth = 1.5)
x = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5']
plt.plot(x, Baseline, alpha=0.8, label="ER", linewidth=3, marker= 'o')
plt.plot(x, Self_drop, alpha=0.8, label="DOA-self", linewidth=3, marker= 'o')
plt.plot(x, gaussian, alpha=0.8, label="Gaussian", linewidth=3, marker= 'o')
plt.plot(x, vMF, alpha=0.8, label="vMF", linewidth=3, marker= 'o')
plt.plot(x, new_drop, alpha=0.8, label="DOA", linewidth=3, marker= 'o')
plt.plot(x, Trans, alpha=0.8, label="VT", linewidth=3, marker= 'o')
plt.plot(x, Adv, alpha=0.8, label="WAP", linewidth=3, marker= 'o')
plt.tick_params(labelsize=14)
ax = plt.gca()
bwith = 1.
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
plt.legend(loc="best", fontsize=14, edgecolor='black')
plt.show()
plt.savefig('./acc_curve_%s.pdf' %title, bbox_inches = 'tight')

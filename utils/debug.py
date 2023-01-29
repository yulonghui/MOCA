
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

def euclid2polar(feat):
    norm_x = torch.norm(feat.clone(), 2, 1, keepdim=True)
    feat = feat / norm_x
    feat_2 = feat * feat
    length = feat.shape[1] - 1
    for idx in range(length):
        sum_ = torch.sum(feat_2[:,idx:], dim=1)
        if idx != length -1:
            polar = torch.arccos(feat[:, idx]/torch.sqrt(sum_)) 
        else:
            polar = torch.arccos(feat[:, idx]/torch.sqrt(sum_)) 
        if idx == 0:
            polar_feat = polar
        else:
            if len(polar_feat.shape) <2:
                polar_feat, polar = polar_feat.unsqueeze(-1), polar.unsqueeze(-1)
            else:
                polar = polar.unsqueeze(-1)
            polar_feat = torch.cat((polar_feat, polar), dim=1)
    return polar_feat

def polar2euclid(feat):
    length = feat.shape[1] + 1
    sin_feat = torch.sin(feat)
    cos_feat = torch.cos(feat)
    
    sin_product = torch.cumprod(sin_feat, dim=1)
    for idx in range(length):
        if idx == 0 :
            euclid = cos_feat[:, 0]
        else:
            if idx != length -1 :
                euclid = sin_product[:, idx -1]
                euclid = euclid * cos_feat[: , idx]
            else:
                euclid = sin_product[:, idx -2]
                euclid = euclid * sin_feat[: , idx-1]
        if idx == 0:
            euclid_feat = euclid
        else:
            if len(euclid_feat.shape) <2:
                euclid_feat, euclid = euclid_feat.unsqueeze(-1), euclid.unsqueeze(-1)
            else:
                euclid = euclid.unsqueeze(-1)
            euclid_feat = torch.cat((euclid_feat, euclid), dim=1)
    
    return euclid_feat
# feat = torch.normal(0.0, 1, size=(32,512))
# print(0,feat[0])
# feat = euclid2polar(feat)
# print(1,feat[0])
# polar_noise = torch.from_numpy(np.random.vonmises(0, 0000000, (feat.shape[0], feat.shape[1])))
# polar_noise.to(feat)
# feat = feat + polar_noise
# print(2,feat[0])
# feat = polar2euclid(feat)
# print(3,feat[0])

def visualize_2d(feat, labels, step):
    feat = feat.numpy()
    plt.figure(figsize=(6, 6))
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[:, 0], feat[:, 1], '.')
    XMax = np.max(feat[:,0]) 
    XMin = np.min(feat[:,1])
    YMax = np.max(feat[:,0])
    YMin = np.min(feat[:,1])

    plt.xlim(xmin=XMin,xmax=XMax)
    plt.ylim(ymin=YMin,ymax=YMax)
    plt.savefig('./%s.pdf' % str(step))

def visualize_3d(feat, labels, step):
    feat = feat.numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')


    ax.scatter(feat[:,0], feat[:,1], feat[:,2])
    plt.show()
    plt.savefig('./%s.pdf' % str(step))
    plt.show()

# feat = torch.from_numpy(np.random.vonmises(0, 1000, (1000, 2)))
# feat = feat
# mask = feat < 0
# copy_feat = copy.deepcopy(feat)
# feat[mask] = -feat[mask]
# # feat[:, :-1] = copy_feat[:, :-1]
# feat = polar2euclid(feat)
# visualize_3d(feat, feat, 'vmf')

# feat2 = euclid2polar(feat)

# feat2 = polar2euclid(feat2)

# visualize_3d(feat2, feat2, 'vmf2')

# true_ = feat == feat2
# print(true_)
# print(feat,feat2)
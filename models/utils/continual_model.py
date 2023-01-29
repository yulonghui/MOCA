# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
import numpy as np
import copy
from torch.nn import functional as F
import math
from torch.optim import SGD
from collections import OrderedDict
EPS = 1E-20

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()
        
        self.current_task = 0
        self.buff_feature = []
        self.buff_labels = []
        self.new_feature = []
        self.new_labels = []
        self.buff_noise = []
        self.EMA_cLass_mean = {}
        self.proxy = copy.deepcopy(self.net)
        self.proxy.to(self.device)
        self.proxy_optim = SGD(self.proxy.parameters(), lr=self.args.lr)
        self.theta_list = []

        self.all_iteration = 0
        for i in range(201):
            self.EMA_cLass_mean[i] = 0.01 + torch.zeros(512).to(self.device)

    def diff_in_weights(self, model, proxy):
        diff_dict = OrderedDict()
        model_state_dict = model.state_dict()
        proxy_state_dict = proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_w.size()) <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
        return diff_dict

    def calc_awp(self, inputs, targets):
        if self.all_iteration > 1000:
            if self.all_iteration % self.args.inner_iter ==0:
                self.proxy.load_state_dict(self.net.state_dict())    
        else:
            self.proxy.load_state_dict(self.net.state_dict())
        self.proxy.train()
        
        if self.args.advloss == 'nega':
            loss = - F.cross_entropy(self.proxy(inputs), targets)
        else:
            loss = F.cross_entropy(self.proxy(inputs), targets)
        loss = self.args.gamma_loss * loss
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = self.diff_in_weights(self.net, self.proxy)
        return diff

    def norm_scale(self, theta, theta_limit):
        theta = torch.arccos(theta)
        theta_limit = torch.tensor(theta_limit/180 * math.pi).to(theta.device)
        norm_scale = torch.sin(theta_limit) / (torch.sin(theta) * torch.cos(theta_limit) - torch.cos(theta) * torch.sin(theta_limit))
        return norm_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
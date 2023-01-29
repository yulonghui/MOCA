# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
######################################### back up ######################################################
import torch
import os
from utils.buffer import Buffer
from utils.debug import euclid2polar, polar2euclid
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import copy
from torch.nn import functional as F
import math
from torch.optim import SGD
from collections import OrderedDict
from utils.vmf_sampling import VonMisesFisher
from utils.mixup_utils import mixup_old_data
import torch.nn as nn
EPS = 1E-20

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

############################################### adv adv adv perturbations ###########################################
############################################### adv adv adv perturbations ###########################################
class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.buff_feature = []
        self.buff_labels = []
        self.new_feature = []
        self.new_labels = []
        self.buff_noise = []
        self.ori_buffer_feat = []
        self.EMA_cLass_mean = {}
        self.proxy = copy.deepcopy(self.net)
        self.proxy.to(self.device)
        self.proxy_optim = SGD(self.proxy.parameters(), lr=self.args.lr)
        self.theta_list = []
        self.args.alpha=0.1
        self.args.beta=0.5
        self.seen_so_far = torch.tensor([]).long().to(self.device)

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
    
    def observe_pp(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        new_labels = labels
        new_features = self.net.feature_forward(inputs)
        new_features.retain_grad()
        outputs = self.net.linear(new_features)
        # outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            buf_features = self.net.feature_forward(buf_inputs)
            buf_features.retain_grad()
            buf_outputs = self.net.linear(buf_features)
            
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()
        self.all_iteration +=1
        if self.current_task > 0:
            if self.all_iteration % 2 ==0:
                if new_labels.shape == buf_labels.shape:
                    grad = buf_features.grad
                    new_grad = new_features.grad
                    self.preserve_feat(buf_features, buf_labels, new_features, new_labels, new_grad, grad)

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def observe_ace(self, inputs, labels, not_aug_inputs):
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        new_features = self.net.feature_forward(inputs)
        new_features.retain_grad()
        logits = self.net.linear(new_features)
        # logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (100 - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)
        new_labels = labels
        self.all_iteration +=1
        if self.current_task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            buf_features = self.net.feature_forward(buf_inputs)
            buf_features.retain_grad()
            logits_re = self.net.linear(buf_features)
            
            loss_re = self.loss(logits_re, buf_labels)

        loss += loss_re

        loss.backward()
        if self.current_task > 0:
            if self.all_iteration % 2 ==0:
                if new_labels.shape == buf_labels.shape:
                    grad = buf_features.grad
                    new_grad = new_features.grad
                    self.preserve_feat(buf_features, buf_labels, new_features, new_labels, new_grad, grad)

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()
    
    def observe_mixup(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        new_inputs = copy.deepcopy(inputs)
        new_labels = copy.deepcopy(labels)
        def mixup_criterion(y_a, y_b, lam):
            return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            
            buf_features = self.net.feature_forward(buf_inputs)
            new_features = self.net.feature_forward(new_inputs)
            
            if buf_labels.shape[0] == new_labels.shape[0] and (self.current_task > 0):
                all_feature = torch.cat((buf_features, new_features.detach()), 0)
                all_labels = torch.cat((buf_labels, new_labels), 0)
                mixed_feat, y_a, y_b, lam = mixup_old_data(all_feature, all_labels, alpha=1.0)
                buf_outputs = self.net.linear(mixed_feat)
                # criterion = nn.CrossEntropyLoss().cuda()
                # loss_func = mixup_criterion(y_a, y_b, lam)
                # loss += loss_func(criterion, buf_outputs)
                loss += self.loss(buf_outputs, buf_labels)
            else:
                buf_outputs = self.net(buf_inputs)
                loss += self.loss(buf_outputs, buf_labels)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    def observe(self, inputs, labels, not_aug_inputs):
        if self.args.method2 == 'ace' :
            return self.observe_ace(inputs, labels, not_aug_inputs)
        if self.args.method2 == 'pp' :
            return self.observe_pp(inputs, labels, not_aug_inputs)
        if 'mixup' in self.args.method2 :
            return self.observe_mixup(inputs, labels, not_aug_inputs)
        real_batch_size = inputs.shape[0]
        self.proxy_optim.param_groups[0]['lr'] = self.opt.param_groups[0]['lr']
        self.opt.zero_grad()
        if self.args.dataset == 'seq-cifar10':
            old_class = self.current_task * 2
        if self.args.dataset == 'seq-cifar100':
            old_class = self.current_task * 20
        if self.args.dataset == 'seq-tinyimg':
            old_class = self.current_task * 20
        new_inputs = copy.deepcopy(inputs)
        new_labels = copy.deepcopy(labels)
        new_features = self.net.feature_forward(new_inputs)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_features = self.net.feature_forward(buf_inputs)
 
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        new_features.retain_grad()
        new_outputs = self.net.linear(new_features)
        loss = self.loss(new_outputs, new_labels)
        
        self.all_iteration +=1
        if not self.buffer.is_empty():
            buf_outputs = self.net.linear(buf_features)
            ori_loss = self.loss(buf_outputs, buf_labels)
            if old_class > 0 :
                buf_features_grad = self.net.feature_forward(buf_inputs)
                buf_features_grad.retain_grad()
                old_index = buf_labels < old_class
                if new_labels.shape == buf_labels.shape:
                    if self.args.noise_type == 'adv': 
                        target_buf_labels = copy.deepcopy(buf_labels)
                        target_buf_labels[:] = old_class
                        if self.args.target_type == 'buf_labels':
                            diff = self.calc_awp(buf_inputs[old_index], buf_labels[old_index])
                        if self.args.target_type == 'target_buf_labels':
                            diff = self.calc_awp(buf_inputs[old_index], target_buf_labels[old_index])
                        if self.args.target_type == 'new_labels':    
                            diff = self.calc_awp(buf_inputs[old_index], new_labels[old_index])
                        if self.args.target_type == 'new_labels_2':
                            target_buf_labels = copy.deepcopy(buf_labels)
                            target_buf_labels[:] = new_labels[0].item()    
                            diff = self.calc_awp(buf_inputs[old_index], target_buf_labels[old_index])
                        buf_features_noise = self.proxy.feature_forward(buf_inputs)
                    else:
                        buf_features_noise = new_features.detach()
                        new_features_noise = new_features.detach()
                        if  'vmf' in self.args.method2:

                            vmf_buf_features = buf_features.detach()
                            norm_x = torch.norm(vmf_buf_features.clone(), 2, 1, keepdim=True)
                            vmf_buf_features = vmf_buf_features / norm_x

                            kappa = self.args.gamma_loss

                            z_var = torch.full([buf_features_noise.shape[0], 1], kappa).to(vmf_buf_features)
                            q_z = VonMisesFisher(vmf_buf_features, z_var)
                            noise_z = q_z.rsample()

                            buf_features_noise = noise_z.detach() - vmf_buf_features.detach()
                        if self.args.method2 == 'class_mean':
                            for name, para in self.net.linear.named_parameters():
                                if name == 'weight':
                                    weight_para = copy.deepcopy(para)
                                target_2_class_mean = new_labels
                                head_mean = weight_para[target_2_class_mean].detach()
                            buf_features_noise = new_features - head_mean
                            buf_features_noise = buf_features_noise.detach()
                        if self.args.method2 == 'drop_self':
                            buf_features_noise = self.net.feature_forward(buf_inputs)
                            buf_features_noise = F.dropout(buf_features_noise, p= 0.5).detach()
                        if self.args.method2 == 'drop_new':
                            buf_features_noise = self.net.feature_forward(new_inputs).detach()
                            buf_features_noise = F.dropout(buf_features_noise, p= 0.5).detach()
                        if self.args.method2 == 'new':
                            buf_features_noise = self.net.feature_forward(new_inputs).detach()
                            buf_features_noise = buf_features_noise.detach()
                        if 'gaussian' in self.args.method2:
                            buf_features_noise = self.net.feature_forward(new_inputs).detach()
                            buf_features_noise = torch.normal(0.0, 1, size=buf_features_noise.shape).to(buf_features_noise.device).detach()
                    
                    # norm_x = torch.norm(buf_features.clone(), 2, 1, keepdim=True)
                    # mean_norm_x = torch.mean(norm_x.detach()).item()
                    # norm_noise = torch.norm(buf_features_noise.clone(), 2, 1, keepdim=True)
                    # mean_norm_noise = torch.mean(norm_noise.detach()).item()

                    if self.args.norm_add == 'norm_add':
                        norm_x = torch.norm(buf_features.clone(), 2, 1, keepdim=True)
                        buf_features = buf_features_grad / norm_x
                        mean_norm_x = torch.mean(norm_x.detach()).item()

                        # buf_features_noise = F.dropout(buf_features_noise, p= 0.5).detach()
                        norm_noise = torch.norm(buf_features_noise.clone(), 2, 1, keepdim=True)
                        buf_features_noise = buf_features_noise / norm_noise
                        mean_norm_noise = torch.mean(norm_noise.detach()).item()
                    
                    ori_buf_features = copy.deepcopy(buf_features.detach())
                    # if self.args.c_theta > 0:
                    #     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    #     ori_buf_features = copy.deepcopy(buf_features.detach())
                    #     cos_theta = cos(buf_features.detach(), buf_features_noise.detach())
                    #     norm_scale = self.norm_scale(cos_theta, self.args.c_theta)
                    #     if self.args.on_sphere == 'on':
                    #         norm_index = norm_scale > 0
                    #     else:
                    #         norm_index = (norm_scale < 1) & (norm_scale > 0)
                    #     buf_features_noise[norm_index] =  buf_features_noise[norm_index] * norm_scale[norm_index].unsqueeze(-1)
                    
                    # self_buf_features_noise = self.net.feature_forward(buf_inputs)
                    # self_buf_features_noise = F.dropout(self_buf_features_noise, p= 0.5).detach()

                    buf_features[old_index] = buf_features[old_index] + self.args.para_scale * buf_features_noise.detach()[old_index]
                    # buf_features = buf_features_grad + self.args.para_scale * buf_features_noise.detach()
                else:
                    print('error')
            buf_features.retain_grad()
            buf_outputs = self.net.linear(buf_features)
            if self.args.para_scale > 0:
                loss +=  self.loss(buf_outputs, buf_labels)
            else:
                buf_features = self.net.feature_forward(buf_inputs)
                buf_features.retain_grad()
                buf_outputs = self.net.linear(buf_features)
                ori_loss = self.loss(buf_outputs, buf_labels)
                loss += ori_loss
        loss.backward()
        if self.current_task >= 4:
            if self.all_iteration % 2 ==0:
                if new_labels.shape == buf_labels.shape:
                    grad = buf_features.grad
                    new_grad = new_features.grad
                    self.preserve_feat(buf_features, buf_labels, new_features, new_labels, new_grad, grad)
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    def norm_scale(self, theta, theta_limit):
        theta = torch.arccos(theta)
        theta_limit = torch.tensor(theta_limit/180 * math.pi).to(theta.device)
        norm_scale = torch.sin(theta_limit) / (torch.sin(theta) * torch.cos(theta_limit) - torch.cos(theta) * torch.sin(theta_limit))
        return norm_scale
    def preserve_feat(self, buff_feats, buff_labels, new_feats, new_labels, ori_buffer_feat, buff_noises=None):
        self.new_feature.append(new_feats.cpu().detach().tolist())
        self.new_labels.append(new_labels.cpu().detach().tolist())
        self.buff_feature.append(buff_feats.cpu().detach().tolist())
        self.ori_buffer_feat.append(ori_buffer_feat.cpu().detach().tolist())
        self.buff_labels.append(buff_labels.cpu().detach().tolist())
        if buff_noises!=None:
            self.buff_noise.append(buff_noises.cpu().detach().tolist())
        
        return None

    def endtask_preserve(self):
        import numpy as np
        self.buff_feature = np.array(self.buff_feature)
        self.new_feature = np.array(self.new_feature)
        self.buff_labels = np.array(self.buff_labels)
        self.new_labels = np.array(self.new_labels)
        self.buff_noise = np.array(self.buff_noise)
        self.ori_buffer_feat = np.array(self.ori_buffer_feat)
        name = str(self.current_task) + '_' + self.args.target_type  + '_' + str(self.args.gamma_loss) + '_' + self.args.noise_type + '_' + self.args.method2 + '_' + str(self.args.c_theta) + '_' + str(self.args.para_scale)
        np.save('output_debug84/buff_featurte_task_%s.npy' %name, self.buff_feature) # 保存为.npy格式
        np.save('output_debug84/new_featurte_task_%s.npy' %name, self.new_feature) # 保存为.npy格式
        np.save('output_debug84/buff_labels_task_%s.npy' %name, self.buff_labels) # 保存为.npy格式
        np.save('output_debug84/new_labels_task_%s.npy' %name, self.new_labels) # 保存为.npy格式
        np.save('output_debug84/buff_noise_task_%s.npy' %name, self.buff_noise) # 保存为.npy格式
        np.save('output_debug84/ori_buffer_feat_task_%s.npy' %name, self.ori_buffer_feat) # 保存为.npy格式
        self.buff_feature = []
        self.buff_labels = []
        self.new_feature = []
        self.new_labels = []
        self.buff_noise = []
        self.ori_buffer_feat = []

    def iteration_preserve(self, iteration):
        import numpy as np
        self.buff_feature = np.array(self.buff_feature)
        self.new_feature = np.array(self.new_feature)
        self.buff_labels = np.array(self.buff_labels)
        self.new_labels = np.array(self.new_labels)
        self.buff_noise = np.array(self.buff_noise)
        self.ori_buffer_feat = np.array(self.ori_buffer_feat)
        name = str(self.current_task) + '_' + self.args.target_type  + '_' + str(self.args.gamma_loss) + '_' + self.args.noise_type + '_' + self.args.method2 + '_' + str(self.args.c_theta) + '_' + str(self.args.para_scale)
        feat_dir = 'output_debug84/' + str(iteration) + '/' 
        np.save(feat_dir + 'buff_featurte_task_%s.npy' %name, self.buff_feature) # 保存为.npy格式
        np.save(feat_dir + 'new_featurte_task_%s.npy' %name, self.new_feature) # 保存为.npy格式
        np.save(feat_dir + 'buff_labels_task_%s.npy' %name, self.buff_labels) # 保存为.npy格式
        np.save(feat_dir + 'new_labels_task_%s.npy' %name, self.new_labels) # 保存为.npy格式
        np.save('output_debug84/buff_noise_task_%s.npy' %name, self.buff_noise) # 保存为.npy格式
        np.save(feat_dir + 'ori_buffer_feat_task_%s.npy' %name, self.ori_buffer_feat) # 保存为.npy格式
        self.buff_feature = []
        self.buff_labels = []
        self.new_feature = []
        self.new_labels = []
        self.buff_noise = []
        self.ori_buffer_feat = []

    def end_task(self, dataset) -> None:
        import numpy as np
        self.theta_list = np.array(self.theta_list)
        self.current_task += 1
        name = str(self.current_task) + '_' + self.args.target_type  + '_' + str(self.args.gamma_loss) + '_' + self.args.noise_type + '_' + self.args.method2 + '_' + str(self.args.c_theta) + '_' + str(self.args.para_scale)
        self.theta_list = []
        if self.current_task >=5:
            self.endtask_preserve()
        name = self.args.target_type  + '_' + str(self.args.gamma_loss) + '_' + self.args.noise_type + '_' + self.args.method2 + '_' + str(self.args.c_theta) + '_' + str(self.args.para_scale)
        model_dir = os.path.join('output_debug84', "task_models", dataset.NAME, name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))
        self.buff_feature = []
        self.buff_labels = []
        self.new_feature = []
        self.new_labels = []
        self.buff_noise = []
        self.ori_buffer_feat = []

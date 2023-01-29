# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from utils.spkdloss import SPKDLoss
from datasets import get_dataset
from torch.nn import functional as F
from utils.args import *
import torch
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.augmentations import *
from utils.batch_norm import bn_track_stats
from utils.simclrloss import SupConLoss

import numpy as np
import copy
from torch.nn import functional as F
import math
from torch.optim import SGD
from collections import OrderedDict
EPS = 1E-20

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--lambd', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--m', type=float, default=0.3)

    parser.add_argument('--simclr_temp', type=float, default=5)
    parser.add_argument('--simclr_batch_size', type=int, default=64)
    parser.add_argument('--simclr_num_aug', type=int, default=2)
    
    return parser

class XDer(ContinualModel):
    NAME = 'xder'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(XDer, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.tasks = get_dataset(args).N_TASKS
        self.task = 0
        self.update_counter = torch.zeros(self.args.buffer_size).to(self.device)
        
        denorm = get_dataset(args).get_denormalization_transform()
        self.dataset_mean, self.dataset_std = denorm.mean, denorm.std
        self.dataset_shape = get_dataset(args).get_data_loaders()[0].dataset.data.shape[2]
        self.gpu_augmentation = strong_aug(self.dataset_shape, self.dataset_mean, self.dataset_std)
        self.simclr_lss = SupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='sum')

        if not hasattr(self.args, 'start_from'):
            self.args.start_from=0

    def end_task(self, dataset):
        self.current_task += 1
        tng = self.training
        self.train()

        if self.args.start_from is None or self.task >= self.args.start_from:
            # Reduce Memory Buffer
            if self.task > 0:
                examples_per_class = self.args.buffer_size // ((self.task + 1) * self.cpt)
                buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                self.buffer.empty()
                for tl in buf_lab.unique():
                    idx = tl == buf_lab
                    ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                    first = min(ex.shape[0], examples_per_class)
                    self.buffer.add_data(
                        examples=ex[:first],
                        labels = lab[:first],
                        logits=log[:first],
                        task_labels=tasklab[:first]
                    )
            
            # Add new task data
            examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
            examples_per_class = examples_last_task // self.cpt
            ce = torch.tensor([examples_per_class] * self.cpt).int()
            ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

            with torch.no_grad():
                with bn_track_stats(self, False):
                    if self.args.start_from is None or self.args.start_from <= self.task:
                        for data in dataset.train_loader:
                            inputs, labels, not_aug_inputs = data
                            inputs = inputs.to(self.device)
                            not_aug_inputs = not_aug_inputs.to(self.device)
                            outputs = self.net(inputs)
                            if all(ce == 0):
                                break

                            # Update past logits
                            if self.task > 0:
                                outputs = self.update_logits(outputs, outputs, labels, 0, self.task)

                            flags = torch.zeros(len(inputs)).bool()
                            for j in range(len(flags)):
                                if ce[labels[j] % self.cpt] > 0:
                                    flags[j] = True
                                    ce[labels[j] % self.cpt] -= 1

                            self.buffer.add_data(examples=not_aug_inputs[flags],
                                                labels=labels[flags],
                                                logits=outputs.data[flags],
                                                task_labels=(torch.ones(len(not_aug_inputs)) *
                                                            (self.task))[flags])

                    # Update future past logits
                    buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.buffer.buffer_size, 
                        transform=self.transform, return_index=True)
                    
                    
                    buf_outputs = []
                    while len(buf_inputs):
                        buf_outputs.append(self.net(buf_inputs[:self.args.batch_size]))
                        buf_inputs = buf_inputs[self.args.batch_size:]
                    buf_outputs = torch.cat(buf_outputs)

                    chosen = (buf_labels // self.cpt) < self.task 
                    
                    if chosen.any():
                        to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.task, self.tasks - self.task)
                        self.buffer.logits[buf_idx[chosen],:] = to_transplant.to(self.buffer.device)
                        self.buffer.task_labels[buf_idx[chosen]] = self.task
                    
        self.task += 1
        self.update_counter = torch.zeros(self.args.buffer_size).to(self.device)

        self.train(tng)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):
        
        transplant = new[:, task_start*self.cpt:(task_start+n_tasks)*self.cpt]
        
        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1,self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1,self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, task_start*self.cpt:(task_start+n_tasks)*self.cpt] = transplant
        
        return old

    def observe(self, inputs, labels, not_aug_inputs):

        
        self.opt.zero_grad()

        outputs = self.net(inputs).float()

        # Present head
        loss_stream = self.loss(outputs[:,self.task*self.cpt:(self.task+1)*self.cpt], labels % self.cpt)

        loss_der, loss_derpp = torch.tensor(0.), torch.tensor(0.)
        if not self.buffer.is_empty():
            # Distillation Replay Loss (all heads)
            buf_idx1, buf_inputs1, buf_labels1, buf_logits1, buf_tl1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_outputs1 = self.net(buf_inputs1).float()

            buf_logits1 = buf_logits1.type(buf_outputs1.dtype)
            mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_der = self.args.alpha * mse.mean()

            # Label Replay Loss (past heads)
            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_outputs2 = self.net(buf_inputs2).float()
            
            if self.args.para_scale > 0:
                variation_ce = self.variation_loss(inputs, labels, not_aug_inputs)
                buf_ce = variation_ce
            else:
                buf_ce = self.loss(buf_outputs2[:, :(self.task)*self.cpt], buf_labels2)
            loss_derpp = self.args.beta * buf_ce

            # Merge Batches & Remove Duplicates
            buf_idx = torch.cat([buf_idx1, buf_idx2])
            buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
            buf_labels = torch.cat([buf_labels1, buf_labels2])
            buf_logits = torch.cat([buf_logits1, buf_logits2])
            buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
            buf_tl = torch.cat([buf_tl1, buf_tl2])
            eyey = torch.eye(self.buffer.buffer_size).to(self.device)[buf_idx]
            umask = (eyey * eyey.cumsum(0)).sum(1) < 2

            buf_idx = buf_idx[umask]
            buf_inputs = buf_inputs[umask]
            buf_labels = buf_labels[umask]
            buf_logits = buf_logits[umask]
            buf_outputs = buf_outputs[umask]
            buf_tl = buf_tl[umask]

            # Update Future Past Logits
            with torch.no_grad():
                chosen = (buf_labels // self.cpt) < self.task     
                self.update_counter[buf_idx[chosen]] += 1
                c = chosen.clone()
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                if chosen.any():
                    assert self.task > 0
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.task, self.tasks - self.task)
                    self.buffer.logits[buf_idx[chosen],:] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.task

        # Consistency Loss (future heads)
        loss_cons, loss_dp = torch.tensor(0.), torch.tensor(0.)
        loss_cons = loss_cons.type(loss_stream.dtype)
        loss_dp = loss_dp.type(loss_stream.dtype)
        if self.task < self.tasks - 1:
            
            scl_labels = labels[:self.args.simclr_batch_size]
            scl_na_inputs = not_aug_inputs[:self.args.simclr_batch_size]
            if not self.buffer.is_empty():
                buf_idxscl, buf_na_inputsscl, buf_labelsscl, buf_logitsscl, _ = self.buffer.get_data(self.args.simclr_batch_size, transform=None, return_index=True)
                scl_na_inputs = torch.cat([buf_na_inputsscl, scl_na_inputs])
                scl_labels = torch.cat([buf_labelsscl, scl_labels])
            with torch.no_grad():
                scl_inputs = self.gpu_augmentation(scl_na_inputs.repeat_interleave(self.args.simclr_num_aug, 0)).to(self.device)
                        
            with bn_track_stats(self, False):
                scl_outputs = self.net(scl_inputs).float()
            
            if self.task < self.tasks - 1:
                scl_featuresFull = scl_outputs.reshape(-1, self.args.simclr_num_aug, scl_outputs.shape[-1]) # [N, n_aug, 100]
                
                scl_features = scl_featuresFull[:, :, (self.task+1)*self.cpt:] # [N, n_aug, 70]
                scl_n_heads = self.tasks - self.task - 1
                
                scl_features = torch.stack(scl_features.split(self.cpt, 2), 1) # [N, 7, n_aug, 10]

                loss_cons = torch.stack([self.simclr_lss(features=F.normalize(scl_features[:,h], dim=2), labels=scl_labels) for h in range(scl_n_heads)]).sum()
                
                loss_cons /= scl_n_heads * scl_features.shape[0]
                loss_cons *= self.args.lambd

        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.task > 0: 
            chead = F.softmax(outputs[:, :(self.task+1)*self.cpt], 1)
        
            good_head = chead[:,self.task*self.cpt:(self.task+1)*self.cpt]
            bad_head  = chead[:,:self.cpt*self.task]
            
            loss_constr = bad_head.max(1)[0].detach() + self.args.m - good_head.max(1)[0]
            
            mask = loss_constr > 0
                
            if (mask).any():
                loss_constr_past = self.args.eta * loss_constr[mask].mean()


        # Future Logits Constraint
        loss_constr_futu = torch.tensor(0.)
        if self.task < self.tasks - 1:
            bad_head = outputs[:,(self.task+1)*self.cpt:]
            good_head = outputs[:,self.task*self.cpt:(self.task+1)*self.cpt]

            if not self.buffer.is_empty():
                buf_tlgt = buf_labels // self.cpt
                bad_head = torch.cat([bad_head, buf_outputs[:,(self.task+1)*self.cpt:]])
                good_head  = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

            loss_constr = bad_head.max(1)[0] + self.args.m - good_head.max(1)[0]
            
            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = self.args.eta * loss_constr[mask].mean()
            
        loss = loss_stream + loss_der + loss_derpp + loss_cons + loss_dp + loss_constr_futu + loss_constr_past
        
        loss.backward()
        self.opt.step()

        return loss.item()

    def variation_loss(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        self.proxy_optim.param_groups[0]['lr'] = self.opt.param_groups[0]['lr']
        self.opt.zero_grad()
        if self.args.dataset != 'seq-cifar10':
            old_class = self.current_task *20
        else:
            old_class = self.current_task *2
        new_inputs = copy.deepcopy(inputs)
        new_labels = copy.deepcopy(labels)
                
        new_features = self.net.feature_forward(new_inputs)
        if not self.buffer.is_empty():
            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_inputs, buf_labels = buf_inputs2, buf_labels2
            buf_features = self.net.feature_forward(buf_inputs)
            new_inputs = copy.deepcopy(inputs)
            new_labels = copy.deepcopy(labels)        
            new_features = self.net.feature_forward(new_inputs)
 
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        loss = 0
        
        self.all_iteration +=1
        if self.all_iteration ==10:
            print('target_buf_labels = copy.deepcopy(buf_labels)target_buf_labels[:] = old_classdiff = self.calc_awp(buf_inputs, target_buf_labels)')
        if not self.buffer.is_empty():
            if old_class > 0 :
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
                        new_features_noise = new_features.detach()
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
                        if self.args.method2 == 'gaussian':
                            buf_features_noise = self.net.feature_forward(new_inputs).detach()
                            buf_features_noise = buf_features_noise.detach()
                    
                    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    
                    norm_x = torch.norm(buf_features.clone(), 2, 1, keepdim=True)
                    mean_norm_x = torch.mean(norm_x.detach()).item()
                    norm_noise = torch.norm(buf_features_noise.clone(), 2, 1, keepdim=True)
                    mean_norm_noise = torch.mean(norm_noise.detach()).item()

                    if self.args.norm_add == 'norm_add':
                        norm_x = torch.norm(buf_features.clone(), 2, 1, keepdim=True)
                        buf_features = buf_features / norm_x
                        mean_norm_x = torch.mean(norm_x.detach()).item()

                        # buf_features_noise = F.dropout(buf_features_noise, p= 0.5).detach()
                        norm_noise = torch.norm(buf_features_noise.clone(), 2, 1, keepdim=True)
                        buf_features_noise = buf_features_noise / norm_noise
                        mean_norm_noise = torch.mean(norm_noise.detach()).item()
                    if self.args.method2 == 'gaussian':
                        buf_features_noise = norm_noise * torch.normal(0.0, 1, size=buf_features_noise.shape).to(buf_features_noise.device)
                        norm_noise = torch.norm(buf_features_noise.clone(), 2, 1, keepdim=True)
                        buf_features_noise = buf_features_noise / norm_noise
                        mean_norm_noise = torch.mean(norm_noise.detach()).item()
                    
                    if self.args.c_theta > 0:
                        ori_buf_features = copy.deepcopy(buf_features.detach())
                        cos_theta = cos(buf_features.detach(), buf_features_noise.detach())
                        norm_scale = self.norm_scale(cos_theta, self.args.c_theta)
                        if self.args.on_sphere == 'on':
                            norm_index = norm_scale > 0
                        else:
                            norm_index = (norm_scale < 1) & (norm_scale > 0)
                        buf_features_noise[norm_index] =  buf_features_noise[norm_index] * norm_scale[norm_index].unsqueeze(-1)
#                         print(norm_scale[norm_index].unsqueeze(-1))
                    if self.args.para_scale > 0:
                        index = int(buf_features[old_index].shape[0]/5)
                        t1 = torch.randint(0, 10, (1,))
                        if t1.item() == 9:
                            pass
                        else:
                            buf_features[old_index] += buf_features_noise.detach()[old_index]
                        
                    cos_theta = cos(buf_features.detach()[old_index], ori_buf_features.detach()[old_index])
                    max_theta, min_theta, mean_theta = max(cos_theta), min(cos_theta), torch.mean(cos_theta)
                    theta_list = [max_theta, min_theta, mean_theta, mean_norm_x, mean_norm_noise]
                    self.theta_list.append(theta_list)
                    # if self.all_iteration % 8 ==0:
                    #     self.preserve_feat(buf_features, buf_labels, new_features, new_labels, buf_features_noise)
                else:
                    print('error')
            buf_outputs = self.net.linear(buf_features)
            loss += self.loss(buf_outputs, buf_labels)
        return loss




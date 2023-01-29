# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
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


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        print('acc', correct, total)
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def evaluate_variance(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    all_feat = None
    all_label = None
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                    feat_ = model.net(inputs, returnt = 'features')
                else:
                    outputs = model(inputs)
                    feat_ = model.net(inputs, returnt = 'features')
                if all_feat == None:
                    all_feat = feat_
                    all_label = labels
                else:
                    all_feat = torch.cat((all_feat, feat_), 0)
                    all_label = torch.cat((all_label, labels), 0)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        print('acc', correct, total)
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    print(all_feat.shape, all_label.shape)
    compute_test_variance(all_feat, all_label)
    model.net.train(status)
    return accs, accs_mask_classes

def compute_test_variance(feat, label):
    angular_list, cos_list = [], []
    for i_class in range(0,100):
        mask = label == i_class
        class_feat = feat[mask] 

        class_feat = F.normalize(class_feat, dim=1, p=2)
        m_feature = torch.mean(class_feat, dim=0).unsqueeze(0)
        m_feature = F.normalize(m_feature, dim=1, p=2)

        angular, cosine_sim = compute_angular_similar(class_feat, m_feature)
        angular_list.append(angular)
        cos_list.append(cosine_sim)
    print(sum(angular_list[:80])/80, sum(angular_list[80:])/20)
    print(sum(cos_list[:80])/80, sum(cos_list[80:])/20)

def compute_angular_similar(class_feat, m_feature):
    RMSSTD = 0
    cosine_sim = 0
    for i_ter in range(class_feat.shape[0]):
        cos_sim = torch.cosine_similarity(m_feature[0], class_feat[i_ter], dim=0)
        cosine_sim += cos_sim
        angular_ = torch.arccos(cos_sim)
        RMSSTD += angular_
    cosine_sim = cosine_sim / (class_feat.shape[0])
    RMSSTD = RMSSTD / (class_feat.shape[0])
    RMSSTD = torch.rad2deg(RMSSTD)
    return RMSSTD.item(), cosine_sim.item()

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
            
            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))


def test_variance(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)
    for t in range(dataset.N_TASKS):
        train_loader, test_loader = dataset.get_data_loaders()
    print(file=sys.stderr)
    name_dict = {
    'er':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_er_45.0_0.0/',
    'vmf':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_vmf_45.0_1.0/',
    'gaussian':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_gaussian_45.0_1.0/',
    'drop_self':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_drop_self_45.0_1.0/',
    'drop_new':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_drop_new_45.0_1.0/',
    'class_mean':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_1.0_noise_class_mean_45.0_1.0/',
    'adv':'./output_testvariance_5task/task_models/seq-cifar100/new_labels_100.0_adv_vmf_45.0_1.5/'
    }
    for t in range(dataset.N_TASKS - 1, dataset.N_TASKS):
        # train_loader, test_loader = dataset.get_data_loaders()
        for item in name_dict.keys():
            name_dir = name_dict[item]
            dirs = name_dir + 'task_5_model.ph'
            model.net = torch.load(dirs)
            model.net.eval()
            accs = evaluate_variance(model, dataset)
            results.append(accs[0])
            results_mask_classes.append(accs[1])

            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
            print(item)




def test_visual(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    EPS = 1E-20
    current_task = '5'
    target_type = 'target_buf_labels'
    gamma_loss = '1.0'
    noise_type = 'noise'
    method2 = 'gaussian'
    c_theta = '45.0'
    para_scale = '1.0'

    s_list = []
    method_list = ['drop_self', 'drop_new', 'class_mean', 'gaussian', 'none']
    for method2 in method_list:
        if method2 == 'none':
            target_type = 'new_labels'
            gamma_loss = '50.0'
            noise_type = 'adv'

        epoch = 50
        name = str(current_task) + '_' + target_type  + \
        '_' + str(gamma_loss) + '_' + noise_type + \
            '_' + method2 + '_' + str(c_theta) +\
            '_' + str(para_scale)
        model_name = target_type  + \
        '_' + str(gamma_loss) + '_' + noise_type + \
            '_' + method2 + '_' + str(c_theta) +\
            '_' + str(para_scale)

        model_path = './output/task_models/seq-cifar100/%s' %model_name + '/task_5_model.ph'
        net = torch.load(model_path)
        net.train()
        feat = np.load('output/buff_featurte_task_%s.npy' %name, allow_pickle=True)
        label = np.load('output/buff_labels_task_%s.npy' %name, allow_pickle=True)
        new_feat = np.load('output/new_featurte_task_%s.npy' %name, allow_pickle=True)
        new_label = np.load('output/new_labels_task_%s.npy' %name, allow_pickle=True)
        ori_feat = np.load('output/ori_buffer_feat_task_%s.npy' %name, allow_pickle=True)
        feat, label, ori_feat, new_feat, new_label = trans_feat(feat, label, ori_feat, new_feat, new_label)
        aug_feat_1, ori_feat_1 = copy.deepcopy(feat), copy.deepcopy(ori_feat)
        feat_noise = aug_feat_1.detach() - ori_feat_1.detach()
        feat, label, feat_noise, new_feat, new_label= feat.cuda(), label.cuda(), feat_noise.cuda(), new_feat.cuda(), new_label.cuda()
        feat_noise, feat, new_feat = feat_noise.to(torch.float32), feat.to(torch.float32), new_feat.to(torch.float32)
        feat.requires_grad = True
        new_feat.requires_grad = True
        if method2 == 'gaussian':
            feat_noise = torch.normal(0.0, 1, size=feat_noise.shape).to(feat_noise.device)
        if method2 != 'none':
            feat2 = feat + feat_noise
        else:
            feat2 = feat + feat_noise
        num_per_epoch = int(feat_noise.shape[0]/50 * 1)
        mask = label[-num_per_epoch:] < 100

        feat_all = torch.cat((feat2, new_feat), 0)
        label_all = torch.cat((label, new_label), 0)


        pred = net.classifier(feat_all)
        loss = F.cross_entropy(pred, label_all)
        loss.backward()
        feat.grad = torch.where(torch.isnan(feat.grad), torch.full_like(feat.grad, 0), feat.grad)

        u, s, v = torch.svd(torch.cat((feat.grad[-num_per_epoch:][mask], new_feat.grad[-num_per_epoch:][mask]), 0))
        s_list.append(s.cpu())

    method_list = ['V-self-drop', 'V-new-drop', 'V-trans', 'V-gaussian', 'V-adv']
    visual_svd(s_list, method_list)



def visual_svd(s_list, method_list):
    x = np.arange(0,512)
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    plt.clf()
    for i in range(len(s_list)):
        s_list[i] = s_list[i] / torch.sum(s_list[i])
    plt.plot(x[:100], s_list[3][:100], label=method_list[3])
    plt.plot(x[:100], s_list[0][:100], label=method_list[0])
    plt.plot(x[:100], s_list[1][:100], label=method_list[1])
    plt.plot(x[:100], s_list[2][:100], label=method_list[2])
    plt.plot(x[:100], s_list[4][:100], label=method_list[4])
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Singular Value', fontsize=14)
    plt.legend()

    ax = plt.gca()
    bwith = 1.
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    plt.tick_params(labelsize=14)
    plt.savefig('./svd2.pdf')
    plt.show

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
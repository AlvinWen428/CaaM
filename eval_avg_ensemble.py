# train.py
#!/usr/bin/env	python3

import os
import random
#debug
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
import argparse
import time
import yaml
from datetime import datetime
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)
from conf import settings
from utils import get_network, get_test_dataloader, get_val_dataloader, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
    update, get_mean_std, Acc_Per_Context, Acc_Per_Context_Class, penalty, cal_acc, get_custom_network, get_custom_network_vit, \
    save_model, load_model, get_parameter_number, init_training_dataloader, get_custom_network_crop_conditioned, \
    get_custom_network_saliency
from train_module import train_env_ours, auto_split, refine_split, update_pre_optimizer, update_pre_optimizer_vit, update_bias_optimizer, auto_cluster
from eval_module import eval_training, eval_best, eval_mode
from timm.scheduler import create_scheduler


class AvgEnsembleModel(nn.Module):
    def __init__(self, net1, net2):
        super(AvgEnsembleModel, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, processed_x):
        logits1 = self.net1(x)
        logits2 = self.net2(processed_x)
        return (logits1 + logits2) / 2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='load the config file')
    parser.add_argument('-net', type=str, default='resnet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-name', type=str, default=None, help='experiment name')
    args = parser.parse_args()

    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)

    config['processed_image_folder'] = '/data/cwen/NICO/crop_by_saliency'

    args.net = config['net']
    training_opt = config['training_opt']
    variance_opt = config['variance_opt']
    exp_name = args.name if args.name is not None else config['exp_name']

    if 'mixup' in training_opt and training_opt['mixup'] == True:
        print('use mixup ...')
    # ============================================================================
    # SEED
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(training_opt['seed'])
    if if_cuda:
        torch.cuda.manual_seed(training_opt['seed'])
        torch.cuda.manual_seed_all(training_opt['seed'])
    random.seed(training_opt['seed'])
    np.random.seed(training_opt['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ============================================================================
    # MODEL
    if variance_opt['mode'] in ['ours']:
        if config['net'] == 'vit':
            vanilla_net = get_custom_network_vit(args, variance_opt)
            saliency_only_net = get_custom_network_vit(args, variance_opt)
        else:
            vanilla_net = get_custom_network(args, variance_opt)
            saliency_only_net = get_custom_network(args, variance_opt)
    elif variance_opt['mode'] in ['chuan']:
        vanilla_net = get_custom_network_crop_conditioned(args, variance_opt)
        saliency_only_net = get_custom_network_crop_conditioned(args, variance_opt)
    elif variance_opt['mode'] in ['chuan_saliency']:
        vanilla_net = get_custom_network_saliency(args, variance_opt)
        saliency_only_net = get_custom_network_saliency(args, variance_opt)
    else:
        vanilla_net = get_network(args)
        saliency_only_net = get_network(args)
    get_parameter_number(vanilla_net)

    load_model(vanilla_net, 'checkpoint/resnet18/baseline_resnet18/resnet18-180-regular.pth')
    load_model(saliency_only_net, 'checkpoint/resnet18/baseline_resnet18_bf0.02_saliency_crop_input/resnet18-180-regular.pth')

    net = AvgEnsembleModel(vanilla_net, saliency_only_net)

    # ============================================================================
    # DATA PREPROCESSING
    if config['dataset'] is not 'Cifar':
        # mean, std = get_mean_std(config['image_folder'])
        mean, std = training_opt['mean'], training_opt['std']
    else:
        mean, std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD

    train_loader_init = init_training_dataloader(config, mean, std, variance_opt['balance_factor'])

    if 'env_type' in variance_opt and variance_opt['env_type'] in ['auto-baseline', 'auto-iter', 'auto-iter-cluster']:
        if variance_opt['env_type'] == 'auto-iter-cluster':
            pre_train_loader, _, __ = train_loader_init.get_pre_dataloader(batch_size=128, num_workers=4, shuffle=False, n_env=variance_opt['n_env'])
        else:
            pre_train_loader, pre_optimizer, pre_schedule = train_loader_init.get_pre_dataloader(batch_size=128, num_workers=4, shuffle=True, n_env=variance_opt['n_env'])
        if variance_opt['from_scratch']:
            pre_split_softmax, pre_split = auto_split(ref_net, pre_train_loader, pre_optimizer, pre_schedule, pre_train_loader.dataset.soft_split)
            np.save('misc/test_unbalance_'+exp_name+'.npy', pre_split.detach().cpu().numpy())
            pre_train_loader.dataset.soft_split = pre_split
            exit()
        else:
            pre_split = np.load('misc/unbalance_nico_resnet18_split.npy')
            pre_split = torch.from_numpy(pre_split).cuda()
            pre_train_loader.dataset.soft_split = torch.nn.Parameter(torch.randn_like(pre_split))
            pre_split_softmax = F.softmax(pre_split, dim=-1)

        pre_split = torch.zeros_like(pre_split_softmax).scatter_(1, torch.argmax(pre_split_softmax, 1).unsqueeze(1), 1)

    else:
        pre_split = None

    if 'resnet' in args.net:
        dim_classifier = 512
    else:
        dim_classifier = 256
    if 'env_type' in variance_opt and variance_opt['env_type'] == 'auto-iter':
        bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
        bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
        bias_dataloader = train_loader_init.get_bias_dataloader(batch_size=128, num_workers=4, shuffle=True)

    val_loader = get_val_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )


    test_loader = get_test_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )

    if variance_opt['mode'] not in ['chuan', 'chuan_saliency']:
        train_loss_function = test_loss_function = nn.CrossEntropyLoss()
    else:
        def two_ce_loss(output, target):
            assert isinstance(output, tuple)
            assert len(output) == 2
            return F.cross_entropy(output[0], target) + F.cross_entropy(output[1], target)
        train_loss_function = two_ce_loss
        test_loss_function = nn.CrossEntropyLoss()

    args.net = 'saliency_conditioned_avg_ensemble'
    val_acc = eval_mode(config, args, net, val_loader, test_loss_function, None)
    test_acc = eval_mode(config, args, net, test_loader, test_loss_function, None)
    print('Val Score: %s  Test Score: %s' %(val_acc.item(), test_acc.item()))

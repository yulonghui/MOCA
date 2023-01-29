# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')

    #############################################################
    parser.add_argument('--noise_std', type=float, default=0.01, required=False,
                        help='noise_std', )
    parser.add_argument('--c_theta', type=float, default=30, required=False,
                        help='c_theta', )
    parser.add_argument('--on_sphere', type=str, default='none', required=False,
                        help='on_sphere.')
    parser.add_argument('--noise_type', type=str, default='adv', required=False,
                        help='noise_type.')
    parser.add_argument('--noise_factor', type=float, default=0.01, required=False,
                        help='noise_factor', )
    parser.add_argument('--drop_rate', type=float, default=0.5, required=False,
                        help='drop_rate', )
    parser.add_argument('--mix_rate', type=float, default=0.5, required=False,
                        help='mix_rate', )

    parser.add_argument('--drop_factor', type=float, default=0.7, required=False,
                        help='drop_factor', )
    parser.add_argument('--gaussian_factor', type=float, default=0.3, required=False,
                        help='gaussian_factor', )

    parser.add_argument('--para_scale', type=float, default=1, required=False,
                        help='para_scale', )
    parser.add_argument('--inner_iter', type=float, default=5, required=False,
                        help='inner_iter', )
    parser.add_argument('--gamma_loss', type=float, default=0.01, required=False,
                        help='gamma', )
    parser.add_argument('--method2', default="mean", type=str,
                        help='Directory where data files are stored.')
    parser.add_argument('--norm_add', default="none", type=str,
                        help='Directory where data files are stored.')
    parser.add_argument('--target_type', default="mean", type=str,
                    help='Directory where data files are stored.')
    parser.add_argument('--advloss', default="nega", type=str,
                help='Directory where data files are stored.')

    parser.add_argument('--optimizer', default="SGD", type=str,
                        help='Directory where data files are stored.')
                    
    parser.add_argument('--epsilon', default= 0.05, type=float,
                        help='Directory where data files are stored.')
    parser.add_argument('--cos_temp', type=float, default=15, required=False,
                    help='cos_temp', )


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')

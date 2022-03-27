from tkinter import E
import numpy as np
import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from params import train_config
from data.utils import create_folder, create_logging
from data.data_loader import AudioDataset, TrainSampler, EvaluateSampler, collate_fn

def train(args):

    # Parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8

    pretrain = True if pretrained_checkpoint_path else False

    hdf5_path = os.path.join(workspace, 'data', 'waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base)
    )
    create_folder(checkpoints_dir)

    logs_dir = os.path.join(workspace, 'logs', filename,
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base)
    )
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Model
    Model = eval(model_type)
    model = Model(train_config.sample_rate, train_config.window_size, train_config.hop_size, train_config.mel_bins,
    train_config.fmin, train_config.fmax, train_config.classes_num, train_config.freeze_base)

    if pretrain:
        logging.info("Load pretrained model from {}".format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info("Load resume model from {}".format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Data
    dataset = AudioDataset()

    # Generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path,
        holdout_fold=holdout_fold,
        batch_size=batch_size
    )

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path,
        holdout_fold=holdout_fold,
        batch_size=batch_size
    )

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    validate_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True
    )

    # Evaluator
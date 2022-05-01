import numpy as np
import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.utils import move_data_to_device

from params import train_config
from data.utils import create_folder, create_logging, get_filename
from data.data_loader import AudioDataset, TrainSampler, EvaluateSampler, collate_fn
from model.models import ConcatCLS
from model.losses import get_loss_func
from model.evaluate import Eva


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
    num_workers = 1
    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base)
    )
    create_folder(checkpoints_dir)

    #logs_dir = os.path.join(workspace, 'logs', filename,
    #    'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
    #    'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
    #    'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base)
    #)
    #create_logging(logs_dir, 'w')
    #logging.info(args)

    # Model
    Model = ConcatCLS # This could be Model = Transfer_Cnn14() in our case, however, here for easy implementation, we will still use this.

    model = Model(train_config.sample_rate, train_config.window_size, train_config.hop_size, train_config.mel_bins,
    train_config.fmin, train_config.fmax, train_config.classes_num, train_config.freeze_base)

    if pretrain:
        print("Load pretrained model from {}".format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        print("Load resume model from {}".format(resume_checkpoint_path))
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
        batch_sampler=validate_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True
    )

    # Evaluator
    evaluator = Eva(model=model)

    train_begin_time = time.time()
    
    loss = nn.BCELoss()
    
    model.train()

    # Train
    print('Start Training')
    for batch_data_dict in train_loader:
                # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        optimizer.zero_grad()
        # Train
        # batch_output_dict = model(batch_data_dict['waveform'], [tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + element.decode("utf-8") + " [SEP]")) for element in batch_data_dict['caption']])
        batch_output_dict = model(batch_data_dict['waveform'], [element for element in batch_data_dict['caption']])

        # for idx in range(len(batch_output_dict)):
        #     print(batch_output_dict[idx], batch_data_dict['target'][idx])
        # loss
        output_loss = loss(batch_output_dict, batch_data_dict['target'])
        
        if iteration % 50 == 0 and iteration > 0:
            print('Iteration Number: {} Loss: {}'.format(iteration, float(output_loss)))

        # Backward
        output_loss.backward()
        optimizer.step()

          # Evaluate
        if iteration % 100 == 0 and iteration > 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                print("-----------------------------------------------")
                print("Iteration: {}".format(iteration))

                train_fin_time = time.time()
                statistics =  evaluator.evaluate(validate_loader)
                print("Validate accuracy: {:.3f}".format(statistics['accuracy']))

                train_time = train_fin_time - train_begin_time
                validate_time = time.time() - train_fin_time



                train_begin_time = time.time()

        # Stop
        if iteration == stop_iteration:
            break

        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--dataset_dir", type=str, required=True, help='Directory of dataset. ')
    parser_train.add_argument("--workspace", type=str, required=True, help='Directory of your workspace. ')
    parser_train.add_argument("--holdout_fold", type=str, choices=['1', '2', '3', '4', '5'], required=True)
    parser_train.add_argument("--model_type", type=str, required=True)
    parser_train.add_argument("--pretrained_checkpoint_path", type=str)
    parser_train.add_argument("--freeze_base", action='store_true', default=False)
    parser_train.add_argument("--loss_type", type=str, required=True)
    parser_train.add_argument("--augmentation", type=str, choices=['none', 'mixup'], required=True) # for easy implementation, I set it to False
    parser_train.add_argument("--learning_rate", type=float, required=True)
    parser_train.add_argument("--batch_size", type=int, required=True)
    parser_train.add_argument("--resume_iteration", type=int)
    parser_train.add_argument("--stop_iteration", type=int, required=True)
    parser_train.add_argument("--cuda", action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception("Args mode should be train. ")

#!bin/bash

DATASET_DIR="/home/user_name/datasets/root"
WORKSPACE="/home/user_name/workspaces/project_name"

python3 data/data_generators.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

PRETRAINED_CHECKPOINT_PATH="/home/user_name/saved_model/Cnn14_mAP=0.431.pth"
# https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1

python3 run_model.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda


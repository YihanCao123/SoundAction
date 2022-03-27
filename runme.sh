#!bin/bash

DATASET_DIR=""
WORKSPACE=""

python3 utils/data_generators.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

PRETRAINED_CHECKPOINT_PATH=""

python3 run_model.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda


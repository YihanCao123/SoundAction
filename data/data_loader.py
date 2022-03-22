""" Load Datasets.
The dataset should be:
root
|--- Class1 (# files)
|--- Class2 (# files)
|--- Class3 (# files)
....
"""
from params import hdf5_config as config
import argparse
import os


def pack_files_to_hdf5(args):
    """Pack .wav files to hdf5 files.
    """
    # Arguments
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    # Paths
    audios_dir = os.path.join(dataset_dir)

    packed_hdf5_path = os.path.join(workspace, "features", "waveform.h5")

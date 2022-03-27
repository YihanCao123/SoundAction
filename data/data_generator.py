""" Load Datasets.
The dataset should be:
root
|--- Class1 (# files)
|--- Class2 (# files)
|--- Class3 (# files)
....
"""
from email import parser
from turtle import shape
from venv import create
from params import hdf5_config as config
import argparse
import os
import time
import numpy as np
import h5py
import librosa

from utils import create_folder, traverse_folder, _convert_float32_to_int16


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len-len(x))))
    else:
        return x[0:max_len]


def pack_audio_files_to_hdf5(args):
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
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)

    audio_names = sorted(audio_names)
    audio_paths = sorted(audio_paths)

    meta_dict = {
        'audio_name': np.array(audio_names),
        'audio_path': np.array(audio_paths),
        'target': np.array([lb_to_idx[audio_name.split('.')[0]] for audio_name in audio_names]),
        'fold': np.arange(len(audio_names)) % 10 + 1 
    }

    audios_num = len(meta_dict['audio_name'])

    feature_time = time.time()

    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S80'
        )

        hf.create_dataset(
            name='waveform',
            shape=(audios_num, clip_samples),
            dtype=np.int16,
        )

        hf.create_dataset(
            name='target',
            shape=(audios_num, classes_num),
            dtype=np.float32,
        )

        hf.create_dataset(
            name='fold',
            shape=(audios_num,),
            dtype=np.int32
        )

        for n in range(audios_num):
            print(n)
            audio_name = meta_dict['audio_name'][n]
            fold = meta_dict['fold'][n]
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            audio = pad_truncate_sequence(audio, clip_samples)

            hf['audio_name'][n] = audio_name.encode()
            hf['waveform'][n] = _convert_float32_to_int16(audio)
            hf['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
            hf['fold'][n] = meta_dict['fold'][n]

    print("Write hdf5 to {}".format(packed_hdf5_path))
    print("Time {:.3f} s".format(time.time() - feature_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument("--dataset_dir", type=str, required=True, help='Directory of dataset. ')
    parser_pack_audio.add_argument("--workspace", type=str, required=True, help='Directory of your workspace. ')
    
    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
    else:
        raise Exception('Incorrect arguments!')

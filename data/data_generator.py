""" Load Datasets.
The dataset should be:
root
|--- Class1 (# files)
|--- Class2 (# files)
|--- Class3 (# files)
....
"""

# importing sys
import sys
  
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/content/SoundAction')
from params import hdf5_config as config
import argparse
import os
import time
import numpy as np
import h5py
import librosa

from utils import create_folder, traverse_folder, _convert_float32_to_int16

import csv

def parse_label_dict(filepath, label_lst):
    label_dict = {}
    with open(filepath) as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            if row[3] in label_lst:
                label_dict[row[0]] = row[3]
    return label_dict


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

    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    fold_dict = config.fold_dict
    av_dict = config.av_dict
    av_length = config.av_length
    openl3_dict = config.openl3_dict
    openl3_shape = config.openl3_shape

    # Paths
    audios_dir = os.path.join(dataset_dir)

    packed_hdf5_path = os.path.join(workspace, "features", "waveform.h5")
    create_folder(os.path.dirname(packed_hdf5_path))

    audio_names, audio_paths = traverse_folder(audios_dir)

    for an in audio_names:
        if an in av_dict.keys():
            pass
        else:
            print('{} is not in the av dict!'.format(an))


    # validation
    # label_dict = parse_label_dict('/content/ESC-50-master/meta/esc50.csv', ['airplane', 'breathing','cat','car_horn'])
    #for i in range(len(audio_names)):
      #print(audio_names[i], audio_paths[i].split('/')[3], lb_to_idx[audio_paths[i].split('/')[3]], label_dict[audio_names[i]] == audio_paths[i].split('/')[3])

    print(audio_names)
    meta_dict = {
        'audio_name': np.array(audio_names),
        'audio_path': np.array(audio_paths),
        'target': np.array([lb_to_idx[audio_path.split('/')[3]] for audio_path in audio_paths]),
        'fold': np.array([fold_dict[audio_name] for audio_name in audio_names]),
        'action_vector': np.array([av_dict[name] for name in audio_names]),
        'openl3_embedding': np.array([openl3_dict[name] for name in audio_names]),
    }
    print(np.array([fold_dict[audio_name] for audio_name in audio_names]))

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
        
        hf.create_dataset(
            name='action_vector',
            shape=(audios_num, av_length),
            dtype=np.float32
        )

        hf.create_dataset(
            name='openl3_embedding',
            shape=(audios_num, openl3_shape[0], openl3_shape[1]),
            dtype=np.float32
        )

        for n in range(audios_num):
            audio_name = meta_dict['audio_name'][n]
            fold = meta_dict['fold'][n]
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            audio = pad_truncate_sequence(audio, clip_samples)

            hf['audio_name'][n] = audio_name.encode()
            hf['waveform'][n] = _convert_float32_to_int16(audio)
            hf['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
            hf['fold'][n] = meta_dict['fold'][n]
            hf['action_vector'][n] = meta_dict['action_vector'][n]
            hf['openl3_embedding'][n] = meta_dict['openl3_embedding'][n]

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

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
import argparse
import os
import time
import numpy as np
import h5py
import librosa
import csv

sys.path.insert(0, '/content/SoundAction')
from params import hdf5_config as config
from utils import create_folder, traverse_folder, _convert_float32_to_int16


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
        return np.concatenate((x, np.zeros(max_len - len(x))))
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
    caption_dict = config.caption_dict
    num_negative = config.num_negaitve

    # Paths
    audios_dir = os.path.join(dataset_dir)

    packed_hdf5_path = os.path.join(workspace, "features", "waveform.h5")
    create_folder(os.path.dirname(packed_hdf5_path))

    audio_names, audio_paths = traverse_folder(audios_dir)


    assert len(audio_names) == len(audio_paths), 'Different Length for audio name and path'

    repeated_audio_name =  np.repeat(np.array(audio_names), 5 + num_negative)
    repeated_audio_path = np.repeat(np.array(audio_paths), 5 + num_negative)
    target_neg = np.array([([1, 1, 1, 1, 1] + [0] * num_negative) for i in range(len(audio_paths))]).reshape(-1)
    fold_np = np.random.permutation(len(target_neg)) % 5
    caption_np = np.array([caption_dict[audio_name] for audio_name in audio_names]).reshape(-1)

    assert len(target_neg) == len(repeated_audio_path), 'Different Length for target and path'
    assert len(target_neg) == len(repeated_audio_name), 'Different Length for audio name and target'
    assert len(target_neg) == len(caption_np), 'Different Length for audio caption and target'
    assert len(target_neg) == len(fold_np), 'Different Length for fold and target'
    audios_num = len(repeated_audio_name)

    print('Validating...')
    for i in range(audios_num):
        caption_lst = caption_dict[repeated_audio_name[i]]
        if caption_np[i] not in caption_lst:
            print('Wrong caption for idx {}'.format(i+1))
        if caption_np[i] not in caption_lst[:5] and target_neg[i] != 0:
            print('Wrong negative target for idx {}'.format(i+1))
        if caption_np[i] in caption_lst[:5] and target_neg[i] == 0:
            print('Wrong positive target for idx {}'.format(i+1))
        if caption_np[i] not in caption_lst:
            print('Wrong caption for idx {}'.format(i+1))


    meta_dict = {
        'audio_name': repeated_audio_name,
        'audio_path': repeated_audio_path,
        'target': target_neg,
        'fold': fold_np,
        'caption': caption_np,
    }



    feature_time = time.time()

    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S200'
        )

        hf.create_dataset(
            name='audio_path',
            shape=(audios_num,),
            dtype='S200'
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
            name='caption',
            shape=(audios_num,),
            dtype='S200'
        )

        energy = 5 + num_negative
        storage = energy
        for n in range(audios_num):
            # print("{} / {} Engergy: {}".format(n+1, audios_num, energy))
            

            audio_name = meta_dict['audio_name'][n]
            audio_path = meta_dict['audio_path'][n]

            if n % 2000 == 0:
                print('{}/{}'.format(n+1, audios_num))

            if energy == storage:
                (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_truncate_sequence(audio, clip_samples)
                int16_audio = _convert_float32_to_int16(audio)
            
            energy -= 1
            if energy == 0:
                energy = storage

            hf['audio_name'][n] = audio_name.encode()
            hf['audio_path'][n] = audio_path.encode()
            hf['waveform'][n] = int16_audio
            hf['target'][n] = meta_dict['target'][n] 
            hf['fold'][n] = meta_dict['fold'][n]
            hf['caption'][n] = meta_dict['caption'][n].encode()
            if hf['audio_path'][n].decode() != meta_dict['audio_path'][n]:
                print('Path******{} path: {}, path: {}'.format(n+1, len(hf['audio_path'][n].decode()), len(meta_dict['audio_path'][n])))
            if hf['caption'][n].decode() != meta_dict['caption'][n]:
                print('Caption******{} caption: {}, caption: {}'.format(n+1, len(hf['caption'][n].decode()), len(meta_dict['caption'][n])))
                print(hf['caption'][n].decode())
                print(meta_dict['caption'][n])
            if hf['audio_name'][n].decode() != meta_dict['audio_name'][n]:
                print('Name******{} name: {}, name: {}'.format(n+1, len(hf['audio_name'][n].decode()), len(meta_dict['audio_name'][n])))


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

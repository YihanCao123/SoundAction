import numpy as np
import h5py
import time
import logging

from utils import _convert_int16_to_float32


class AudioDataset:
    """ Creates a dataset object. """
    def __init__(self):
        """This class takes the meta of an audio clip as input and return
        the waveform and target of the audio clip. This class is used by DataLoader.
        Args:
            clip_samples: int
            classes_num: int
        """
        pass

    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
            meta: {
                'audio_name': str,
                'hdf5_path': str,
                'index_in_hdf5': int
            }
        Returns:
            data_dict: {
                'audio_name': str,
                'waveform': (clip_samples,),
                'target': (classes_num,)
            }
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = _convert_int16_to_float32(hf['waveform'][index_in_hdf5])
            target = hf['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target
        }

        return data_dict


class Base:
    """ Creates a dataset object. """
    def __init__(self, indexes_hdf5_path, batch_size, random_seed):
        """Base class of train sampler.
        Args:
            indexes_hdf5_path: string
            batch_size: int
            random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            self.folds = hf['fold'][:].astype(np.float32)

        (self.audios_num, self.classes_num) = self.targets.shape

        logging.info("Training number: {}".format(self.audios_num))
        logging.info("Load target time: {:.3f} s".format(time.time() - load_time))


class TrainSampler:
    def __init__(self, hdf5_path, holdout_fold, batch_size, random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        with h5py.File(hdf5_path, 'r') as hf:
            self.folds = hf['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds != int(holdout_fold))[0]
        self.audios_num = len(self.indexes)

        # Shuffle indexes
        self.random_state.shuffle(self.indexes)

        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training.

        Returns: 
            batch_meta: [
                {
                    'audio_name': 'abcdefg.wav',
                    'hdf5_path': 'xx/balanced_train.h5',
                    'index_in_hdf5': 15734,
                    'target': [0, 1, 0, 0, ....],
                },
                ......
            ]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)

                batch_meta.append({
                    'hdf5_path': self.hdf5_path,
                    'index_in_hdf5': self.indexes[self.pointer]
                })

                i += 1
        
        yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return state

    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class EvaluateSampler:
    def __init__(self, hdf5_path, holdout_fold, batch_size, random_seed=1234):
        self.hdf5_path = hdf5_path
        self.batch_size=  batch_size

        with h5py.File(hdf5_path, 'r') as hf:
            self.folds = hf['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds == int(holdout_fold))[0]
        self.audios_num = len(self.indexes)

    def __iter__(self):
        """Generate batch meta for training. """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.batch_size:
            batch_indexes = np.arange(
                pointer, min(pointer + batch_size, self.audios_num)
            )

            batch_meta = []

            for i in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path,
                    'index_in_hdf5': self.indexes[i]
                })

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.
    Args: 
        list_data_dict: [
            {
                'audio_name': str,
                'waveform': (clip_samples,), ....
            },
            {
                'audio_name': str,
                'waveform': (clip_samples,), ....
            }
        ]

    Returns:
        np_data_dict: {
            'audio_name': (batch_size,),
            'waveform': (batch_size, clip_samples), ....
        }
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
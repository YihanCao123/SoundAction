""" All Params could be used in training and testing.
1. hdf5_config
"""

import csv

def parse_labels(filepath):
    subset = set()
    with open(filepath) as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            subset.add(row[3])
    return sorted(list(subset))

LABELS = parse_labels('/content/ESC-50-master/meta/esc50.csv')

#['airplane', 'breathing','cat','car_horn']


class hdf5_config:
    """HDF5 configs used in data_loader.py. """
    sample_rate = 32000
    clip_samples = sample_rate * 5
    classes_num = len(LABELS)
    lb_to_idx = {lb: idx for idx, lb in enumerate(LABELS)}
    idx_to_lb = {idx: lb for idx, lb in enumerate(LABELS)}


class train_config:
    """Training configs used in run_model.py. """
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    classes_num = len(LABELS)
    freeze_base = True

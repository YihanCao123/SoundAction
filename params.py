""" All Params could be used in training and testing.
1. hdf5_config
"""

import csv

from random import randint
def generate_negative(caption_dict, number_neg):
    key_lst = list(caption_dict.keys())
    key_len = len(key_lst)
    correct_range = len(caption_dict[key_lst[0]])
    for idx in range(key_len):
        for _ in range(number_neg):
            neg_idx = randint(0, key_len - 1)
            while neg_idx == idx:
                neg_idx = randint(0, key_len - 1)
            caption_dict[key_lst[idx]].append(caption_dict[key_lst[neg_idx]][randint(0, correct_range-1)])
    return caption_dict

# def parse_labels(filepath):
#     subset = set()
#     fold_dict = {}
#     with open(filepath) as f:
#         f_csv = csv.reader(f)
#         next(f_csv)
#         for row in f_csv:
#             subset.add(row[3])
#             fold_dict[row[0]] = row[1]
#     return fold_dict, sorted(list(subset))
#
# FOLD_DICT, LABELS = parse_labels('/content/ESC-50-master/meta/esc50.csv')

def parse_caption(filepath):
    fold_dict = {}
    with open(filepath) as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            fold_dict[row[0]] = [row[1], row[2], row[3], row[4], row[5]]
    return fold_dict


#['airplane', 'breathing','cat','car_horn']

NUM_NEG = 2

FOLD_DICT, LABELS = [], [1]
CAPTION_DICT = generate_negative(parse_caption('/content/clotho_captions_development.csv'), NUM_NEG)
class hdf5_config:
    """HDF5 configs used in data_loader.py. """
    sample_rate = 41400
    clip_samples = sample_rate * 5
    classes_num = len(LABELS)
    lb_to_idx = {lb: idx for idx, lb in enumerate(LABELS)}
    idx_to_lb = {idx: lb for idx, lb in enumerate(LABELS)}
    fold_dict = FOLD_DICT
    caption_dict = CAPTION_DICT
    num_negaitve = NUM_NEG


class train_config:
    """Training configs used in run_model.py. """
    sample_rate = 41400
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    classes_num = 2
    freeze_base = True

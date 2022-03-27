""" All Params could be used in training and testing.
1. hdf5_config
"""

LABELS = []


class hdf5_config:
    """HDF5 configs used in data_loader.py. """
    sample_rate = 32000
    clip_samples = sample_rate * 30
    classes_num = len(LABELS)
    lb_to_idx = {lb: idx for idx, lb in enumerate(LABELS)}
    idx_to_lb = {idx: lb for idx, lb in enumerate(LABELS)}


class train_config:
    """Training configs used in run_model.py. """
    sample_rate = 32000
    window_size = 0
    hop_size = 0
    mel_bins = 0
    fmin = 0
    fmax = 0
    classes_num = 0
    freeze_base = True
    

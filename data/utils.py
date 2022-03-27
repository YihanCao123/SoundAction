import os
import logging
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
    
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        formaat='%{asctime}s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datafmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def traverse_folder(fd):
    paths = []
    names = []
    for root, _, files in os.walk(fd):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
    
    return names, paths


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def _convert_float32_to_int16(x):
    """Convert from float32 to int16. """
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)


def _convert_int16_to_float32(x):
    """Convert from int16 to float32. """
    return (x / 32767.).astype(np.float32)
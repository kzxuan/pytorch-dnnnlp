#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple fuctions (file, time, numpy)
Last update: KzXuan, 2019.08.13
"""
import numpy as np


def file_in_dir(path, delete_start=['.']):
    """Get part of the files in path.

    Args:
        path [str]: search path
        delete_start [list]: delete files that start with some symbols

    Returns:
        file_list [list]: list of all the files
    """
    import os

    file_list = os.listdir(path)
    file_list = [ele for ele in file_list if ele[0] not in delete_start]
    return file_list


def time_difference(begin, end, form="%a %b %d %H:%M:%S %z %Y"):
    """Calculate the time difference.

    Args:
        begin [str]: time string
        end [str]: time string
        form [str]: time structure

    Returns:
        seconds [int]: time difference in seconds
    """
    import datetime as dt

    atime = dt.datetime.strptime(begin, form)
    btime = dt.datetime.strptime(end, form)
    return (btime - atime).seconds


def time_str_stamp(str_time, form="%a %b %d %H:%M:%S %z %Y"):
    """Convert time string to stamp number.

    Args:
        str_time [str]: time string
        form [str]: time structure

    Returns:
        time_stamp [int]: unix time stamp
    """
    import time

    time_array = time.strptime(str_time, form)
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def print_shape(data_dict):
    """Print shape of matrix in data_dict.

    Args:
        data_dict [dict]: data dict including matrix
    """
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print("- {} shape: {}".format(key, value.shape))


def one_hot(arr, n_class=0):
    """Change labels to one-hot expression.

    Args:
        arr [np.array]: numpy array
        n_class [int]: number of class

    Returns:
        oh [np.array]: numpy array with one-hot expression
    """
    if arr is None:
        return None
    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        arr = np.array(arr)
        ishape = arr.shape
        arr = arr.flatten()

        n_class = arr.max() + 1 if n_class == 0 else n_class
        assert n_class >= arr.max() + 1, ValueError("Value of 'n_class' is too small.")

        oh = np.zeros((arr.size, n_class), dtype=int)
        oh[np.arange(arr.size), arr] = 1
        oh = np.reshape(oh, (*ishape, -1))

    return oh


def remove_zero_rows(arr):
    """Remove rows with all zero from an matrix.

    Args:
        arr [np.array]: matrix with size (-1, n_class)

    Returns:
        result [np.array]: after removement
        nonzero_row_indice [np.array]: index of nonzero rows
    """
    assert arr.ndim == 2, "Size error of 'arr'."
    nonzero_row_indice, _ = arr.nonzero()
    nonzero_row_indice = np.unique(nonzero_row_indice)
    result = arr[nonzero_row_indice]

    return result, nonzero_row_indice


def mod_fold(length, fold=10):
    """Use mod index to fold.

    Args:
        length [int]: length of data
        fold [int]: fold in need (default=10)

    Returns:
        indexs [list]: [(fold_num, train_index, test_index),]
    """
    indexs = []
    _all = np.arange(length, dtype=int)
    for f in range(fold):
        test = np.arange(f, length, fold, dtype=int)
        train = np.setdiff1d(_all, test)
        indexs.append((f + 1, train, test))
    return indexs


def order_fold(length, fold=10):
    """Use order index to fold.

    Args:
        length [int]: length of data
        fold [int]: fold in need (default=10)

    Returns:
        indexs [list]: [(fold_num, train_index, test_index),]
    """
    indexs = []
    _all = np.arange(length, dtype=int)
    gap, left = length // fold, length % fold
    begin = 0
    for f in range(fold):
        step = gap + 1 if f < left else gap
        test = np.arange(begin, begin + step, dtype=int)
        train = np.setdiff1d(_all, test)
        begin += step
        indexs.append((f + 1, train, test))
    return indexs


def len_to_mask(seq_len, max_seq_len):
    """Convert seq_len to mask matrix.

    Args:
        seq_len [tensor]: sequence length vector (batch_size,)
        max_seq_len [int]: max sequence length

    Returns:
        mask [tensor]: mask matrix (batch_size * max_seq_len)
    """
    if isinstance(seq_len, np.ndarray):
        query = np.arange(0, max_seq_len)
        mask = (query < seq_len.reshape(-1, 1)).astype(int)
    else:
        import torch
        query = torch.arange(0, max_seq_len, device=seq_len.device).float()
        mask = torch.lt(query, seq_len.unsqueeze(1)).int()
    return mask


def mask_to_len(mask):
    """Convert mask matrix to seq_len.

    Args:
        mask [tensor]: mask matrix (batch_size * max_seq_len)

    Returns:
        seq_len [tensor]: sequence length vector (batch_size,)
    """
    if isinstance(mask, np.ndarray):
        seq_len = np.sum(mask, axis=1).astype(int)
    else:
        import torch
        seq_len = torch.sum(mask, dim=1).int()
    return seq_len

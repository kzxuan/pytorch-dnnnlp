#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for operating folder and file or batch
Last update: KzXuan, 2018.12.08
"""
import os
import json
import codecs as cs
from copy import deepcopy
from utils.step_print import slash

sl = slash()


def file_in_dir(now_dir, delete_start=['.']):
    """
    Get all files in dir
    Parameters:
    * delete_start [list]: delete files that start with some symbols
    """
    dir_list = os.listdir(now_dir)
    dir_list = [ele for ele in dir_list if ele[0] not in delete_start]
    return dir_list


def load_list(in_dir, state=None, code='utf-8', line_split=False):
    """
    Load a file that contains lists
    * line_split [bool]: each line contains one element
    """
    if state is not None:
        sl.start("* Load %s" % state)
    data_list = []
    with cs.open(in_dir, 'r', code) as inobj:
        if line_split:
            for line in inobj:
                ele = json.loads(line)
                data_list.append(ele)
        else:
            data_list = json.load(inobj)
    if state is not None:
        sl.stop()
    return data_list


def batch_append(lists, values):
    """
    List batch append
    * lists [list]: batch list including several lists
    * values [list]: list including values that need to be added
    """
    for i in range(len(lists)):
        lists[i].append(values[i])


def batch_index(n_samples, batch_size):
    """
    Get index of each batch
    * n_samples [int]: sum of samples
    - indexs [list]: list of index slices
    """
    indexs = []
    for start in range(0, n_samples, batch_size):
        if start + batch_size <= n_samples:
            indexs.append(slice(start, start + batch_size))
        else:
            indexs.append(slice(start, n_samples))
    return indexs


def one_hot(arr, n_class=0):
    """
    Change labels to one hot model
    """
    import numpy as np

    if arr is None:
        return None
    if n_class == 0:
        n_class = arr.max() + 1
    oh = np.zeros((arr.size, n_class), dtype=int)
    oh[np.arange(arr.size), arr] = 1
    return oh


def print_shape(data_dict):
    """
    Print shape of matrix in data_dict
    * data_dict [dict]: dict value is array
    """
    for key, value in data_dict.items():
        if type(value).__name__ == "ndarray":
            print("- {} shape: {}".format(key, value.shape))
        elif type(value).__name__ == "list":
            if type(value[0]).__name__ == "ndarray":
                for i, ele in enumerate(value):
                    if type(ele).__name__ == "ndarray":
                        print("- {}{} shape: {}".format(key, i, ele.shape))
            elif type(value[0]).__name__ == "list":
                ele = deepcopy(value)
                shape = tuple()
                while type(ele).__name__ == "list":
                    shape = shape + (len(ele),)
                    ele = ele[0]
                print("- {} shape: {}".format(key, shape))


def list_mean(value):
    """
    Mean value of a list
    * value [list]: some numbers
    - return [float]: mean value of a list
    """
    return sum(value) / len(value)


def format_dict(d, key_sep=', ', value_sep=' ', value_format=':.4f'):
    """
    Convert result dict to string
    * d [dict]: including key 'P'/'R'/'F'/'Acc'
    - str_d [string]: string expression
    """
    str_d = key_sep.join(("{}{}{%s}" % value_format).format(
        key, value_sep, value) for (key, value) in d.items()
    )
    return str_d

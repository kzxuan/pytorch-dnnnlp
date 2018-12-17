#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for operating folder and file or batch
Last update: KzXuan, 2018.12.13
"""
import os
import json
import numpy as np
import codecs as cs
from copy import deepcopy
from step_print import slash

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


def output_list(data_list, out_dir, state=None, code='utf-8', line_split=False):
    """
    Output a list with json
    * line_split [bool]: each line contains one element
    """
    if state is not None:
        sl.start("* Output %s" % state)
    with cs.open(out_dir, 'w', code) as outobj:
        if line_split:
            for ele in data_list:
                outobj.write(json.dumps(ele) + '\n')
        else:
            json.dump(data_list, outobj)
    if state is not None:
        sl.stop()
    return 1


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


def output_dict(data_dict, out_dir, state=None, code='utf-8', line_split=False):
    """
    Output a dict with json
    * line_split [bool]: each line contains one element or not
    """
    if state is not None:
        sl.start("* Output %s" % state)
    with cs.open(out_dir, 'w', code) as outobj:
        if line_split:
            for key, value in data_dict.items():
                outobj.write(str(key) + ' ' + json.dumps(value) + '\n')
        else:
            json.dump(data_dict, outobj)
    if state is not None:
        sl.stop()
    return 1


def load_dict(in_dir, state=None, code='utf-8', line_split=False):
    """
    Read a data file that contains dicts
    * line_split [bool]: each line contains one element or not
    """
    if state is not None:
        sl.start("* Load %s" % state)
    data_dict = {}
    with cs.open(in_dir, 'r', code) as inobj:
        if line_split:
            for line in inobj:
                key, value = line.split(' ', 1)
                value = json.loads(value)
                data_dict[key] = value
        else:
            data_dict = json.load(inobj)
    if state is not None:
        sl.stop()
    return data_dict


def read_words(in_dir, state=None, code='utf-8'):
    """
    Read word txt that each line has a word and output a list
    """
    if state is not None:
        sl.start("* Read %s" % state)
    with cs.open(in_dir, 'r', code) as fobj:
        words = [line.split()[0] for line in fobj]
    if state is not None:
        sl.stop()
    return words


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


def time_difference(begin, end, format="%a %b %d %H:%M:%S %z %Y"):
    """
    Calculate the time difference between two tweets
    * begin [str]: time string
    * end [str]: time string
    """
    import datetime as dt
    atime = dt.datetime.strptime(begin, format)
    btime = dt.datetime.strptime(end, format)
    return (btime - atime).seconds


def list_mean(value):
    """
    Mean value of a list
    * value [list]: some numbers
    - return [float]: mean value of a list
    """
    return sum(value) / len(value)


def format_dict(d, key_sep=', ', value_sep=' ', digits=4):
    """
    Convert result dict to string
    * d [dict]: including key 'P'/'R'/'F'/'Acc'
    - str_d [string]: string expression
    """
    for key, value in d.items():
        if type(value).__name__[:5] == "float":
            d[key] = round(value, digits)
    str_d = key_sep.join(
        [("{}{}{}").format(key, value_sep, str(value)) for (key, value) in d.items()]
    )
    return str_d


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


def one_hot(arr, n_class=0):
    """
    Change labels to one hot model
    """
    if arr is None:
        return None
    if n_class == 0:
        n_class = arr.max() + 1
    oh = np.zeros((arr.size, n_class), dtype=int)
    oh[np.arange(arr.size), arr] = 1
    return oh


def remove_zero_rows(array):
    """
    Remove rows with all zero from an matrix
    * array [np.array]: matrix with size (-1, n_class)
    - result [np.array]: after removement
    - nonzero_row_indice [np.array]: index of nonzero rows
    """
    assert array.ndim == 2, "! Wrong size for input."
    nonzero_row_indice, _ = array.nonzero()
    nonzero_row_indice = np.unique(nonzero_row_indice)
    result = array[nonzero_row_indice]
    return result, nonzero_row_indice

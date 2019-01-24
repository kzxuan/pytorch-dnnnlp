#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some base modules for deep neural network
Ubuntu 16.04 & PyTorch 1.0
Last update: KzXuan, 2019.01.23
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import easy_function as ef
import torch.utils.data as Data
import torch.nn.functional as F


def default_args(data_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", default=1, type=int, help="number of GPUs for running")
    parser.add_argument("--data_shuffle", default=False, type=bool, help="shuffle data")
    parser.add_argument("--GRU_enable", default=True, type=bool, help="use LSTM or GRU")
    parser.add_argument("--bi_direction", default=True, type=bool, help="bi direction choice")
    parser.add_argument("--n_layer", default=1, type=int, help="hidden layer number")
    parser.add_argument("--use_attention", default=True, type=bool, help="use attention or not")

    parser.add_argument("--emb_type", default=None, type=str, help="embedding type")
    if data_dict is not None:
        parser.add_argument("--emb_dim", default=data_dict['x'].shape[-1], type=int, help="embedding dimension")
        parser.add_argument("--n_class", default=data_dict['y'].shape[-1], type=int, help="classify classes number")
        parser.add_argument("--n_hierarchy", default=len(data_dict['len']), type=int, help="RNN hierarchy number")
    else:
        parser.add_argument("--emb_dim", default=300, type=int, help="embedding dimension")
        parser.add_argument("--n_class", default=2, type=int, help="classify classes number")
        parser.add_argument("--n_hierarchy", default=1, type=int, help="RNN hierarchy number")

    parser.add_argument("--n_hidden", default=50, type=int, help="hidden layer nodes")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--l2_reg", default=1e-6, type=float, help="l2 regularization")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--iter_times", default=30, type=int, help="iteration times")
    parser.add_argument("--display_step", default=2, type=int, help="display inteval")
    parser.add_argument("--drop_prob", default=0.1, type=float, help="drop out ratio")
    parser.add_argument("--score_standard", default='Acc', type=str, help="score standard")
    args = parser.parse_args()
    return args


class base(object):
    def __init__(self, args):
        """
        Initilize the model data
        * n_gpu [int]: number of GPUs for running
        * data_shuffle [bool]: shuffle data
        * GRU_enable [bool]: use LSTM or GRU model
        * bi_direction [bool]: use bi-direction model or not
        * n_layer [int]: hidden layer number
        * use_attention [bool]: use attention or not
        * emb_type [str]: use None/'const'/'variable'/'random'
        * emb_dim [int]: embedding dimension
        * n_class [int]: number of object classify classes
        * n_hierarchy [int]: number of RNN hierarchies
        * n_hidden [int]: number of hidden layer nodes
        * learning_rate [float]: learning rate
        * l2_reg [float]: L2 regularization parameter
        * batch_size [int]: train batch size
        * iter_times [int]: iteration times
        * display_step [int]: the interval iterations to display
        * drop_prob [float]: drop out ratio
        * score_standard [str]: use 'Ma-P'/'C1-R'/'C1-F'/'Acc'
        """
        self.n_gpu = args.n_gpu
        self.data_shuffle = args.data_shuffle
        self.GRU_enable = args.GRU_enable
        self.bi_direction = args.bi_direction
        self.n_layer = args.n_layer
        self.use_attention = args.use_attention
        self.emb_type = args.emb_type
        self.emb_dim = args.emb_dim
        self.n_class = args.n_class
        self.n_hierarchy = args.n_hierarchy
        self.n_hidden = args.n_hidden
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        self.iter_times = args.iter_times
        self.display_step = args.display_step
        self.drop_prob = args.drop_prob
        self.score_standard = args.score_standard

    def attributes_from_dict(self, args):
        """
        Set attributes name and value from dict
        * args [dict]: dict including name and value of parameters
        """
        for name, value in args.items():
            setattr(self, name, value)

    def create_data_loader(self, *data):
        """
        Create data loader for pytorch
        * data [tensor]: several tensors with the same shape[0]
        - loader [DataLoader]: torch data generator
        """
        dataset = Data.TensorDataset(*data)
        loader = Data.DataLoader(
            dataset=dataset,
            shuffle=self.data_shuffle,
            batch_size=self.batch_size
        )
        return loader

    @staticmethod
    def mod_fold(length, fold=10):
        """
        Use index for fold
        * length [int]: length of data
        * fold [int]: fold in need
        - indexs [list]: [(fold_num, train_index, test_index),]
        """
        indexs = []
        for f in range(fold):
            test = [ind for ind in range(f, length, fold)]
            train = [ind for ind in range(length) if ind not in test]
            indexs.append((f + 1, train, test))
        return indexs

    @staticmethod
    def vote_sequence(predict, ids):
        """
        Vote for predict/label if the sequence is repetitive
        * predict [np.array]: (sample_num, max_seq_len, n_class)
        * ids [list]: ids of each sequence
        - get_pred [np.array]: predict array
        """
        predict = np.argmax(predict, axis=2)
        id_sort = [series[-1] for series in ids]
        id_predict = {id: [] for id in id_sort}

        for i, series in enumerate(ids):
            for j, id in enumerate(series):
                id_predict[id].append(predict[i][j])
        for id in id_predict.keys():
            id_predict[id] = np.argmax(np.bincount(id_predict[id]))

        get_pred = np.array([id_predict[id] for id in id_sort])
        return get_pred

    def average_several_run(self, run, times=5, **run_params):
        """
        Get average result after several running
        * run [function]: model run function which returns a result dict including 'P'/'R'/'F'/'Acc'
        * times [int]: run several times for average
        * run_params [parameter]: some parameters for run function including 'fold'/'verbose'
        """
        results = {}

        for i in range(times):
            print("* Run round: {}".format(i + 1))
            result = run(**run_params)
            for key, score in result.items():
                results.setdefault(key, [])
                results[key].append(score)
            print("*" * 88)
        for key in results:
            results[key] = ef.list_mean(results[key])
        print("* Average score after {} rounds: {} {:6.4f}".format(
              times, self.score_standard, results[self.score_standard]))

    def grid_search(self, run, params_search, **run_params):
        """
        * run [function]: model run function which returns a result dict including 'P'/'R'/'F'/'Acc'
        * params_search [dict]: the argument value need to be tried
        * run_params [parameter]: some parameters for run function including 'fold'/'verbose'
        """
        from sklearn.model_selection import ParameterGrid

        params_search = list(ParameterGrid(params_search))
        results, max_score = {}, -1

        for params in params_search:
            self.attributes_from_dict(params)
            print("* Now params: {}".format(str(params)))
            result = run(**run_params)
            score = result[self.score_standard]
            results[str(params)] = score
            if score > max_score:
                max_score = score
                max_result = result.copy()
                max_score_params = params
            print("- Result: {} {:6.4f}".format(self.score_standard, score))
            print("*" * 88)
        print(
            "* Grid search best:\n  {}\n  {}".format(
                ef.format_dict(max_score_params, value_sep=': '),
                ef.format_dict(max_result)
            )
        )
        print("* All results:\n{}".format(ef.format_dict(results, key_sep='\n', value_sep=': ')))

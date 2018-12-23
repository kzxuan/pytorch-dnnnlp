#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution functions for deep neural models
Ubuntu 16.04 & PyTorch 1.0
Last update: KzXuan, 2018.12.23
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import easy_function as ef
import torch.utils.data as Data
import torch.nn.functional as F
from copy import deepcopy
from dnn.pytorch import base, model
from step_print import table_print, percent
from predict_analysis import predict_analysis


class exec(base.base):
    def __init__(self, data_dict, args=None, class_name=None):
        """
        Initilize execution fuctions
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * args [dict]: all model arguments
        * class_name [list]: name of each class
        """
        args = default_args(data_dict) if args is None else args
        base.base.__init__(self, args)

        self.data_dict = data_dict
        self.class_name = class_name
        self._init_display()

    def _init_display(self):
        """
        Initilize display
        """
        self.prf = self.score_standard.split('-')[0] if self.score_standard != 'Acc' else 'Ma'
        self.col = ["Step", "Loss", "%s-P" % self.prf, "%s-R" % self.prf, "%s-F" % self.prf, "Acc", "Correct"]
        max_width = np.reshape(self.data_dict['y'], [-1, self.n_class]).shape[0]
        data_scale = (len(str(max_width)) + 1) * self.n_class + 1
        self.width = [4, 6, 6, 6, 6, 6, data_scale]

    def _run_train(self, train_loader, **model_params):
        """
        Run train part
        * train_loader [DataLoader]: train data generator
        * model_params [parameter]: more parameters for model
        - losses [float]: loss of one iteration
        """
        self.model.train()
        losses = 0.0
        for step, (x, y, *lq) in enumerate(train_loader):
            if self.cuda_enable:
                x, y, lq = x.cuda(), y.cuda(), [ele.cuda() for ele in lq]
            pred = self.model(x, *lq, **model_params)
            loss = - torch.sum(y.float() * torch.log(pred)) / torch.sum(lq[-1]).float()
            losses += loss.cpu().data.numpy()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        losses = losses / (step + 1)
        return losses

    def _run_test(self, test_loader, **model_params):
        """
        Run test part
        * test_loader [DataLoader]: test data generator
        * model_params [parameter]: more parameters for model
        - preds [np.array]: predicts of the test data
        - tys [np.array]: true label of the test data
        """
        self.model.eval()
        preds, tys = torch.FloatTensor(), torch.LongTensor()
        for step, (tx, ty, *tlq) in enumerate(test_loader):
            if self.cuda_enable:
                tx, ty, tlq = tx.cuda(), ty.cuda(), [ele.cuda() for ele in tlq]
            pred = self.model(tx, *tlq, **model_params)

            preds = torch.cat((preds, pred.cpu()))
            tys = torch.cat((tys, ty.cpu()))
        return preds.data.numpy(), tys.data.numpy()

    def _run(self, now_data_dict, verbose):
        """
        Model run part
        * now_data_dict [dict]: data dict with torch.tensor in it
        * verbose [int]: visual hierarchy including 0/1/2
        - best_iter [int]: the iteration number of the best result
        - best_ty [np.array] the true label of the best result
        - best_pred [np.array]: the prediction of the best result
        - best_result [dict]: the best result value including 'P'/'R'/'F'/'Acc'
        """
        train_loader = self.create_data_loader(
            now_data_dict['x'], now_data_dict['y'], *now_data_dict['len']
        )
        test_loader = self.create_data_loader(
            now_data_dict['tx'], now_data_dict['ty'], *now_data_dict['tlen']
        )

        self.model.load_state_dict(self.model_init)
        best_score = -1
        if verbose > 1:
            ptable = table_print(self.col, self.width, sep="vertical")
        if verbose == 1:
            per = percent("* Run model", self.iter_times)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader)
            pred, ty = self._run_test(test_loader)
            result = predict_analysis(ty, pred, one_hot=True, simple=True, get_prf=self.prf)
            if it % self.display_step == 0 and verbose > 1:
                ptable.print_row(dict(result, **{"Step": it, "Loss": loss}))
            if verbose == 1:
                per.change()

            if result[self.score_standard] > best_score:
                best_score = result[self.score_standard]
                best_result = result.copy()
                best_iter = it
                best_ty = deepcopy(ty)
                best_pred = deepcopy(pred)
        return best_iter, best_ty, best_pred, best_result

    def train_test(self, verbose=2):
        """
        Run train and test
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        now_data_dict = {
            'x': torch.FloatTensor(self.data_dict['x']),
            'tx': torch.FloatTensor(self.data_dict['tx']),
            'y': torch.LongTensor(self.data_dict['y']),
            'ty': torch.LongTensor(self.data_dict['ty']),
            'len': [torch.IntTensor(ele) for ele in self.data_dict['len']],
            'tlen': [torch.IntTensor(ele) for ele in self.data_dict['tlen']]
        }

        best_iter, best_ty, best_pred, best_result = self._run(now_data_dict, verbose)
        one_hot = True if best_ty.ndim > 1 else False
        if verbose > 1:
            ana = predict_analysis(
                best_ty, best_pred, one_hot=one_hot, class_name=self.class_name, simple=False
            )
            print("- Best test result: Iteration {}\n".format(best_iter), ana)
        else:
            print("- Best result: It {:2d}, {}".format(best_iter, ef.format_dict(best_result)))
        return best_result

    def train_itself(self, verbose=2):
        """
        Run test by train data
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        self.data_dict.update({
            'tx': self.data_dict['x'].copy(),
            'ty': self.data_dict['y'].copy(),
            'tlen': self.data_dict['len'].copy()
        })
        return self.train_test(verbose)

    def cross_validation(self, fold=10, verbose=2):
        """
        Run cross validation
        * fold [int]: k fold control
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        kf_results = {}

        for count, train, test in self.mod_fold(self.data_dict['x'].shape[0], fold=fold):
            now_data_dict = {
                'x': torch.FloatTensor(self.data_dict['x'][train]),
                'tx': torch.FloatTensor(self.data_dict['x'][test]),
                'y': torch.LongTensor(self.data_dict['y'][train]),
                'ty': torch.LongTensor(self.data_dict['y'][test]),
                'len': [torch.IntTensor(ele[train]) for ele in self.data_dict['len']],
                'tlen': [torch.IntTensor(ele[test]) for ele in self.data_dict['len']]
            }

            if verbose > 0:
                _ty = np.reshape(now_data_dict['ty'].numpy(), [-1, self.n_class])
                state = np.bincount(np.argmax(ef.remove_zero_rows(_ty)[0], -1))
                print("* Fold {}: {}".format(count, state))
            best_iter, best_ty, best_pred, best_result = self._run(now_data_dict, verbose)
            if verbose > 0:
                print("- Best result: It {:2d}, {}".format(best_iter, ef.format_dict(best_result)))
                print("-" * 88)

            for key, value in best_result.items():
                kf_results.setdefault(key, [])
                kf_results[key].append(value)
        kf_mean = {key: ef.list_mean(value) for key, value in kf_results.items()}
        print("* Avg: {}".format(ef.format_dict(kf_mean)))
        return kf_mean


class CNN_classify(exec):
    def __init__(self, data_dict, emb_matrix=None, args=None,
                 kernel_widths=[1, 2, 3], class_name=None):
        """
        Initilize the CNN classification model
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * emb_matrix [np.array]: word embedding matrix (need emb_type!=None)
        * args [dict]: all model arguments
        * kernel_widths [list]: kernel_widths [list]: list of kernel widths for cnn kernel
        * class_name [list]: name of each class
        """
        exec.__init__(self, data_dict, args, class_name)

        self.model = model.CNN_model(emb_matrix, args, kernel_widths)
        if self.cuda_enable:
            self.gpu_dist = range(self.n_gpu)
            self.model.cuda(self.gpu_dist[0])
            if self.n_gpu > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_dist)
        self.model_init = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )


class RNN_classify(exec):
    def __init__(self, data_dict, emb_matrix=None, args=None, class_name=None):
        """
        Initilize the RNN classification model
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * emb_matrix [np.array]: word embedding matrix (need emb_type!=None)
        * args [dict]: all model arguments
        * class_name [list]: name of each class
        """
        exec.__init__(self, data_dict, args, class_name)

        self.model = model.RNN_model(emb_matrix, args, mode='classify')
        if self.cuda_enable:
            self.gpu_dist = range(self.n_gpu)
            self.model.cuda(self.gpu_dist[0])
            if self.n_gpu > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_dist)
        self.model_init = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )


class RNN_sequence(exec):
    def __init__(self, data_dict, emb_matrix=None, args=None,
                 vote=False, class_name=None):
        """
        Initilize the RNN sequencial labeling model
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * emb_matrix [np.array]: word embedding matrix (need emb_type!=None)
        * args [dict]: all model arguments
        * vote [bool]: vote for duplicate data (need 'id'/'tid' in data_dict)
        * class_name [list]: name of each class
        """
        self.vote = vote
        exec.__init__(self, data_dict, args, class_name)

        self.model = model.RNN_model(emb_matrix, args, mode='sequence')
        if self.cuda_enable:
            self.gpu_dist = range(self.n_gpu)
            self.model.cuda(self.gpu_dist[0])
            if self.n_gpu > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_dist)
        self.model_init = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

    def _run(self, now_data_dict, verbose):
        """
        Model run part
        * now_data_dict [dict]: data dict with torch.tensor in it
        * verbose [int]: visual hierarchy including 0/1/2
        - best_iter [int]: the iteration number of the best result
        - best_ty [np.array] the true label of the best result
        - best_pred [np.array]: the prediction of the best result
        - best_result [dict]: the best result value including 'P'/'R'/'F'/'Acc'
        """
        train_loader = self.create_data_loader(
            now_data_dict['x'], now_data_dict['y'], *now_data_dict['len']
        )
        test_loader = self.create_data_loader(
            now_data_dict['tx'], now_data_dict['ty'], *now_data_dict['tlen']
        )

        self.model.load_state_dict(self.model_init)
        best_score = -1
        if verbose > 1:
            ptable = table_print(self.col, self.width, sep="vertical")
        if verbose == 1:
            per = percent("* Run model", self.iter_times)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader)
            pred, ty = self._run_test(test_loader)

            if self.vote:
                get_pred = self.vote_sequence(pred, self.data_dict['tid'])
                get_ty = self.vote_sequence(ty, self.data_dict['tid'])
            else:
                pred, ty = np.reshape(pred, [-1, self.n_class]), np.reshape(ty, [-1, self.n_class])
                get_ty, nonzero_ind = ef.remove_zero_rows(ty)
                get_pred = pred[nonzero_ind]
                get_ty, get_pred = np.argmax(get_ty, -1), np.argmax(get_pred, -1)

            result = predict_analysis(get_ty, get_pred, one_hot=False, simple=True, get_prf=self.prf)
            if it % self.display_step == 0 and verbose > 1:
                ptable.print_row(dict(result, **{"Step": it, "Loss": loss}))
            if verbose == 1:
                per.change()

            if result[self.score_standard] > best_score:
                best_score = result[self.score_standard]
                best_result = result.copy()
                best_iter = it
                best_ty = deepcopy(get_ty)
                best_pred = deepcopy(get_pred)
        return best_iter, best_ty, best_pred, best_result
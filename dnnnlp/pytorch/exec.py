#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifyution functions for deep neural models.
Last update: KzXuan, 2019.08.12
"""
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import dnnnlp.utils.predict_eval as pe
import dnnnlp.utils.easy_function as ef
from copy import deepcopy
from . import layer, model
from dnnnlp.utils.display_tool import table

logger = logging.getLogger("dnnnlp.pytorch.classify")


def default_args():
    """Set default arguments.

    Returns:
        args [argparse.Namespace]: default arguments which can be pri-set in shell
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", default=1, type=int, help="number of GPUs for running")
    parser.add_argument("--space_turbo", default=True, type=bool, help="use more space to fasten")
    parser.add_argument("--data_shuffle", default=True, type=bool, help="shuffle data")
    parser.add_argument("--emb_type", default=None, type=str, help="embedding type")
    parser.add_argument("--emb_dim", default=300, type=int, help="embedding dimension")
    parser.add_argument("-c", "--n_class", default=2, type=int, help="classify classes number")

    parser.add_argument("--n_hidden", default=50, type=int, help="hidden layer nodes")
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("-l2", "--l2_reg", default=1e-6, type=float, help="l2 regularization")
    parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("-i", "--iter_times", default=30, type=int, help="iteration times")
    parser.add_argument("--display_step", default=2, type=int, help="display inteval")
    parser.add_argument("--drop_prob", default=0.1, type=float, help="drop out ratio")
    parser.add_argument("--eval_metric", default='acc', type=str, help="evaluation metric")

    args = parser.parse_args()
    return args


class exec(object):
    def __init__(self, args):
        """Initilize the model argument.

        Args:
            args [argparse.Namespace]: set arguments including:
                n_gpu [int]: number of GPUs for running
                space_turbo [bool]: use more space to fasten
                data_shuffle [bool]: shuffle data
                emb_type [str]: use None/'const'/'variable'/'random'
                emb_dim [int]: embedding dimension
                n_class [int]: number of object classify classes
                n_hidden [int]: number of hidden layer nodes
                learning_rate [float]: learning rate
                l2_reg [float]: L2 regularization parameter
                batch_size [int]: train batch size
                iter_times [int]: iteration times
                display_step [int]: the interval iterations to display
                drop_prob [float]: drop out ratio
                eval_metric [str]: use 'macro'/'class1'/'acc'/...
        """
        self.args = args
        self.n_gpu = args.n_gpu
        self.space_turbo = args.space_turbo
        self.data_shuffle = args.data_shuffle
        self.emb_type = args.emb_type
        self.emb_dim = args.emb_dim
        self.n_class = args.n_class
        self.n_hidden = args.n_hidden
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        self.iter_times = args.iter_times
        self.display_step = args.display_step
        self.drop_prob = args.drop_prob
        self.eval_metric = args.eval_metric

    def attributes_from_dict(self, args):
        """Set attributes' name and value from dict.

        Args:
            args [dict]: dict including name and value of parameters
        """
        for name, value in args.items():
            setattr(self, name, value)

    def create_data_loader(self, *data):
        """Create data loader for pytorch.

        Args:
            data [tensor]: several tensors with the same shape[0]

        Returns:
            loader [DataLoader]: torch data generator
        """
        dataset = Data.TensorDataset(*data)
        loader = Data.DataLoader(
            dataset=dataset,
            shuffle=self.data_shuffle,
            batch_size=self.batch_size
        )
        return loader


class Classify(exec):
    def __init__(self, model, args, train_x, train_y, train_mask, test_x=None,
                 test_y=None, test_mask=None, class_name=None, device_id=0):
        """Initilize classification method.

        Args:
            model [nn.Module]: a standart pytorch model
            args [dict]: all model arguments
            train_x [np.array/tensor]: training data
            train_y [np.array/tensor]: training label
            train_mask [np.array/tensor]: training mask
            test_x [np.array/tensor]: testing data
            test_y [np.array/tensor]: testing label
            test_mask [np.array/tensor]: testing mask
            class_name [list]: name of each class
            device_id [int]: CPU device for -1, and GPU device for 0/1/...
        """
        exec.__init__(self, args)

        self.class_name = class_name
        self.device_id = device_id
        self.device = torch.device(device_id) if self.n_gpu and self.space_turbo else torch.device("cpu")
        self.model = model
        self._model_initilize()

        self.train_x = torch.as_tensor(train_x, dtype=torch.float, device=self.device)
        self.train_y = torch.as_tensor(train_y, dtype=torch.long, device=self.device)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.int, device=self.device)
        if test_x:
            self.test_x = torch.as_tensor(test_x, dtype=torch.float, device=self.device)
        if test_y:
            self.test_y = torch.as_tensor(test_y, dtype=torch.long, device=self.device)
        if test_mask:
            self.test_mask = torch.as_tensor(test_mask, dtype=torch.int, device=self.device)

        em = "macro" if self.eval_metric == "acc" else self.eval_metric
        self.display_col = ["iter", "loss"] + [em + e for e in ['-p', '-r', '-f']] + ["acc", "correct"]

    def _model_initilize(self):
        """Given initilize fuction for model.
        """
        assert torch.cuda.device_count() >= self.n_gpu, "Not enough GPU devices."
        assert torch.cuda.device_count() >= self.device_id + self.n_gpu, "Not enough GPU devices."
        if self.n_gpu:
            self.model.to(self.device_id)
            if self.n_gpu > 1:
                self.gpu_dist = range(self.device_id, self.device_id + self.n_gpu)
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_dist)

        self.model_init = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )
        self.loss_function = nn.NLLLoss()

    def _run_train(self, train_loader, **model_params):
        """Training part.

        Args:
            train_loader [DataLoader]: train data generator
            model_params [parameter]: more parameters for model

        Returns:
            losses [float]: loss of one iteration
        """
        self.model.train()

        loss = 0.0
        for step, (x, y, m) in enumerate(train_loader):
            if not self.space_turbo and self.n_gpu:
                x, y, m = x.to(self.device_id), y.to(self.device_id), m.to(self.device_id)
            pred = self.model(x, m, **model_params)
            batch_loss = self.loss_function(pred, y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.cpu().data.numpy()
        loss = loss / (step + 1)
        return loss

    def _run_test(self, test_loader, **model_params):
        """Testing part.

        Args:
            test_loader [DataLoader]: test data generator
            model_params [parameter]: more parameters for model

        Returns:
            preds [np.array]: predicts of the test data
            tys [np.array]: true label of the test data
        """
        self.model.eval()

        preds, tys = torch.FloatTensor(), torch.LongTensor()
        for step, (tx, ty, tm) in enumerate(test_loader):
            if not self.space_turbo and self.n_gpu:
                tx, ty, tm = tx.to(self.device_id), ty.to(self.device_id), tm.to(self.device_id)
            pred = self.model(tx, tm, **model_params)

            preds = torch.cat((preds, pred.cpu()))
            tys = torch.cat((tys, ty.cpu()))

        preds = np.argmax(preds.data.numpy(), axis=-1)
        return preds, tys.data.numpy()

    def train_test(self):
        """Run training and testing.

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        assert self.test_x and self.test_y and self.test_mask, ValueError("Test x or y or mask may not exist.")

        train_loader = self.create_data_loader(
            self.train_x, self.train_y, self.train_mask
        )
        test_loader = self.create_data_loader(
            self.test_x, self.test_y, self.test_mask
        )

        results = []
        ptable = table(self.display_col)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader)
            pred, ty = self._run_test(test_loader)

            evals = pe.prfacc1d(ty, pred, one_hot=False)
            evals.update({'iter': it, 'loss': loss})
            if it % self.display_step == 0:
                ptable.row(evals)
            results.append(evals)

        best_evals = pe.maximum(*results, eval_metric=self.eval_metric)
        ptable.line()
        best_evals['iter'] = 'BEST'
        ptable.row(best_evals)
        return best_evals

    def train_itself(self):
        """Run testing with training data.

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        self.test_x = self.train_x
        self.test_y = self.train_y
        self.test_mask = self.train_mask
        return self.train_test()

    def cross_validation(self, fold=10):
        """Run cross validation.

        Args:
            fold [int]: k fold control

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        kf_avg = []
        for count, train, test in ef.mod_fold(self.train_x.shape[0], fold=fold):
            train_loader = self.create_data_loader(
                self.train_x[train], self.train_y[train], self.train_mask[train]
            )
            test_loader = self.create_data_loader(
                self.train_x[test], self.train_y[test], self.train_mask[test]
            )

            print("* Fold {}, label {}".format(count, self.train_y[test].bincount().cpu().numpy()))
            self.model.load_state_dict(self.model_init)

            results = []
            ptable = table(self.display_col)
            for it in range(1, self.iter_times + 1):
                loss = self._run_train(train_loader)
                pred, ty = self._run_test(test_loader)

                evals = pe.prfacc1d(ty, pred, one_hot=False)
                evals.update({'iter': it, 'loss': loss})
                if it % self.display_step == 0:
                    ptable.row(evals)
                results.append(evals)

            best_evals = pe.maximum(*results, eval_metric=self.eval_metric)
            ptable.line()
            ptable.row(dict(best_evals, **{"iter": 'BEST'}))
            print()

            kf_avg.append(best_evals)

        avg_evals = pe.average(*kf_avg)
        ptable = table(['avg'] + self.display_col[1:])
        ptable.row(avg_evals)
        return avg_evals

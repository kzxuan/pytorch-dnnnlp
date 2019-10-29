#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifyution functions for deep neural models.
Last update: KzXuan, 2019.10.29
"""
import time
import torch
import dnnnlp
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.multiprocessing as mp
from copy import deepcopy
from . import utils, layer, model


def default_args():
    """Set default arguments.

    Returns:
        args [argparse.Namespace]: default arguments which can be pri-set in shell
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", default=1, type=int, help="number of GPUs for running")
    parser.add_argument("--space_turbo", default=True, type=bool, help="use more space to fasten")
    parser.add_argument("--rand_seed", default=100, type=int, help="random seed")
    parser.add_argument("--data_shuffle", default=True, type=bool, help="shuffle data")
    parser.add_argument("--emb_type", default=None, type=str, help="embedding type")
    parser.add_argument("--emb_dim", default=300, type=int, help="embedding dimension")
    parser.add_argument("-c", "--n_class", default=2, type=int, help="classify classes number")

    parser.add_argument("--n_hidden", default=50, type=int, help="hidden layer nodes")
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("-l2", "--l2_reg", default=1e-6, type=float, help="l2 regularization")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-i", "--iter_times", default=30, type=int, help="iteration times")
    parser.add_argument("--display_step", default=2, type=int, help="display inteval")
    parser.add_argument("--drop_prob", default=0.1, type=float, help="drop out ratio")
    parser.add_argument("--eval_metric", default='accuracy', type=str, help="evaluation metric")

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
        self.rand_seed = args.rand_seed
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
        self.set_seed(self.rand_seed)
        if self.eval_metric == 'micro':
            self.eval_metric = 'accuracy'

    def set_seed(self, seed):
        """Set random seed.

        Args:
            seed [int]: seed number
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def attributes_from_dict(self, args):
        """Set attributes' name and value from dict.

        Args:
            args [dict]: dict including name and value of parameters
        """
        for name, value in args.items():
            setattr(self, name, value)

    def create_data_loader(self, *data, shuffle=True):
        """Create data loader for pytorch.

        Args:
            data [tensor]: several tensors with the same shape[0]
            shuffle [bool]: data shuffling

        Returns:
            loader [DataLoader]: torch data generator
        """
        dataset = Data.TensorDataset(*data)
        loader = Data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.batch_size
        )
        return loader


class Classify(exec):
    def __init__(self, model, args, train_x, train_y, train_mask,
                 test_x=None, test_y=None, test_mask=None):
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
        """
        exec.__init__(self, args)

        self.model = model

        self.train_x = torch.as_tensor(train_x, dtype=torch.float)
        self.train_y = torch.as_tensor(train_y, dtype=torch.long)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.int)
        self.test_x = test_x if test_x is None else torch.as_tensor(test_x, dtype=torch.float)
        self.test_y = test_y if test_y is None else torch.as_tensor(test_y, dtype=torch.long)
        self.test_mask = test_mask if test_mask is None else torch.as_tensor(test_mask, dtype=torch.int)

    def _model_initilize(self, device_id):
        """Given initilize fuction for model.

        Args:
            device_id [int]: CPU device for -1, and GPU device for 0/1/...
        """
        assert torch.cuda.device_count() >= self.n_gpu, "Not enough GPU devices."
        self.device_id = device_id
        if self.device_id == -1:
            self.n_gpu = 0
            self.device = torch.device('cpu')
        elif self.n_gpu and self.space_turbo:
            self.device = torch.device(self.device_id)
        else:
            self.device = torch.device('cpu')

        # set cuda and parallel
        assert torch.cuda.device_count() >= self.device_id + self.n_gpu, "Not enough GPU devices."
        if self.n_gpu:
            self.model.to(self.device_id)
            if self.n_gpu > 1:
                self.gpu_dist = range(self.device_id, self.device_id + self.n_gpu)
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_dist)

        # get model initial parameters
        self.model_init = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )
        self.loss_function = nn.NLLLoss()

        # move data to initial device
        if self.n_gpu and self.space_turbo:
            self.train_x = self.train_x.to(self.device_id)
            self.train_y = self.train_y.to(self.device_id)
            self.train_mask = self.train_mask.to(self.device_id)
            if self.test_x is not None:
                self.test_x = self.test_x.to(self.device_id)
                self.test_y = self.test_y.to(self.device_id)
                self.test_mask = self.test_mask.to(self.device_id)

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
            evals [dict]: dict of all the evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            preds = torch.FloatTensor()
            for tx, ty, tm in test_loader:
                if not self.space_turbo and self.n_gpu:
                    tx, ty, tm = tx.to(self.device_id), ty.to(self.device_id), tm.to(self.device_id)
                pred = self.model(tx, tm, **model_params)

                preds = torch.cat((preds, pred.cpu().data))
            preds = preds.argmax(dim=-1)

        evals = utils.prfacc(self.test_y, preds, one_hot=False)
        return evals

    def train_test(self, device_id=0, **model_params):
        """Run training and testing.

        Args:
            device_id [int]: CPU device for -1, and GPU device for 0/1/...
            model_params [parameter]: more parameters for model

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        assert self.test_x is not None, ValueError("Test data may not exist.")
        self._model_initilize(device_id)

        train_loader = self.create_data_loader(
            self.train_x, self.train_y, self.train_mask, shuffle=self.data_shuffle
        )
        test_loader = self.create_data_loader(
            self.test_x, self.test_y, self.test_mask, shuffle=False
        )
        self.model.load_state_dict(self.model_init)

        results = []
        ptable = utils.display_prfacc(self.eval_metric, verbose=2)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader, **model_params)
            evals = self._run_test(test_loader, **model_params)

            evals.update({'iter': it, 'loss': loss})
            if it % self.display_step == 0:
                ptable.row(evals)
            results.append(evals)

        best_evals = utils.maximum_prfacc(*results, eval_metric=self.eval_metric)
        ptable.line()
        ptable.row(dict(best_evals, **{"iter": 'BEST'}))
        return best_evals

    def train_itself(self, device_id=0, **model_params):
        """Run testing with training data.

        Args:
            device_id [int]: CPU device for -1, and GPU device for 0/1/...
            model_params [parameter]: more parameters for model

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        self.test_x = self.train_x
        self.test_y = self.train_y
        self.test_mask = self.train_mask
        return self.train_test(device_id, **model_params)

    def cross_validation(self, fold=10, device_id=0, **model_params):
        """Run cross validation.

        Args:
            fold [int]: k fold control
            device_id [int]: CPU device for -1, and GPU device for 0/1/...
            model_params [parameter]: more parameters for model

        Returns:
            best_evals [dict]: the best evaluation metrics
        """
        self._model_initilize(device_id)
        kf_avg = []

        for count, train, test in utils.mod_fold(self.train_x.shape[0], fold=fold):
            self.test_x, self.test_y, self.test_mask = self.train_x[test], self.train_y[test], self.train_mask[test]
            train_loader = self.create_data_loader(
                self.train_x[train], self.train_y[train], self.train_mask[train], shuffle=self.data_shuffle
            )
            test_loader = self.create_data_loader(
                self.test_x, self.test_y, self.test_mask, shuffle=False
            )

            if dnnnlp.verbose.check(2):
                print("\nFold {}".format(count))
            self.model.load_state_dict(self.model_init)

            results = []
            ptable = utils.display_prfacc(self.eval_metric, verbose=2)
            for it in range(1, self.iter_times + 1):
                loss = self._run_train(train_loader, **model_params)
                evals = self._run_test(test_loader, **model_params)

                evals.update({'iter': it, 'loss': loss})
                if it % self.display_step == 0:
                    ptable.row(evals)
                results.append(evals)

            best_evals = utils.maximum_prfacc(*results, eval_metric=self.eval_metric)
            ptable.line()
            ptable.row(dict(best_evals, **{"iter": 'BEST'}))

            kf_avg.append(best_evals)

        avg_evals = utils.average_prfacc(*kf_avg)
        if dnnnlp.verbose.check(1):
            print()
        ptable = utils.display_prfacc(self.eval_metric, verbose=1)
        ptable.row(dict(avg_evals, **{"iter": 'AVG'}))
        return avg_evals


class SequenceLabeling(Classify):
    def __init__(self, model, args, train_x, train_y, train_mask,
                 test_x=None, test_y=None, test_mask=None):
        """Initilize sequence labeling method.

        Args:
            model [nn.Module]: a standart pytorch model
            args [dict]: all model arguments
            train_x [np.array/tensor]: training data
            train_y [np.array/tensor]: training label
            train_mask [np.array/tensor]: training mask
            test_x [np.array/tensor]: testing data
            test_y [np.array/tensor]: testing label
            test_mask [np.array/tensor]: testing mask
        """
        Classify.__init__(
            self, model, args, train_x, train_y, train_mask, test_x, test_y, test_mask
        )

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
            batch_loss = self.model(x, m, y, **model_params)

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
            evals [dict]: dict of all the evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            preds = torch.LongTensor()
            for tx, ty, tm in test_loader:
                if not self.space_turbo and self.n_gpu:
                    tx, ty, tm = tx.to(self.device_id), ty.to(self.device_id), tm.to(self.device_id)
                pred = self.model(tx, tm, **model_params)

                preds = torch.cat((preds, pred.cpu().data))

        evals = utils.prfacc(self.test_y, preds, self.test_mask, one_hot=False)
        return evals


def _device_count():
    return torch.cuda.device_count()


def average_several_run(run_func, args, n_times=4, n_paral=2, **run_params):
    """Get average result after several running.

    Args:
        run_func [func]: running function like 'Classify.train_test'
        args [dict]: all model arguments
        n_times [int]: run several times for average
        n_paral [int]: number of parallel processes
        run_params [dict]: parameters for runing function

    Returns:
        max_socres [dict]: dict of the maximum scores after n_times running
    """
    dnnnlp.verbose.config(0)

    assert not n_times % n_paral, ValueError("'n_times' should be an integral multiple of 'n_paral'.")

    pool = mp.Pool(processes=1)
    check = pool.apply_async(_device_count)
    device_count = check.get()

    scores, processes = [], []
    pool = mp.Pool(processes=n_paral)

    if args.n_gpu > 0:
        assert n_paral * args.n_gpu <= device_count, "Not enough GPU devices."

    for t in range(n_times):
        if args.n_gpu > 0:
            run_params['device_id'] = (t % n_paral) * args.n_gpu
        else:
            run_params['device_id'] = -1

        processes.append(pool.apply_async(
            run_func,
            kwds=run_params.copy()
        ))

        if (t + 1) % n_paral == 0:
            for i, p in enumerate(processes):
                result = p.get()
                scores.append(result)

                print()
                ptable = utils.display_prfacc(args.eval_metric, verbose=0)
                ptable.row(dict(result, **{"iter": t + 2 - len(processes) + i}))
            processes.clear()

    avg_scores = utils.average_prfacc(*scores)
    print()
    ptable = utils.display_prfacc(args.eval_metric, verbose=0)
    ptable.row(dict(avg_scores, **{"iter": 'AVG'}))

    dnnnlp.verbose.config(2)
    return avg_scores


def _pack_run(exec_class, run_func, **run_params):
    time.sleep(1)
    params_set = run_params.pop('params_set')
    exec_class.attributes_from_dict(params_set)
    results = run_func(**run_params)
    return results


def grid_search(exec_class, run_func, args, params_search, n_paral=2, **run_params):
    """Do parameters' grid search.

    Args:
        exec_func [exec]: an exec class like 'Classify'
        run_func [func]: running function like 'Classify.train_test'
        args [dict]: all model arguments
        params_search [dict]: grid search parameters
        n_paral [int]: number of parallel processes
        run_params [dict]: parameters for run function

    Returns:
        max_socres [dict]: dict of the maximum scores after n_times running
    """
    from sklearn.model_selection import ParameterGrid

    dnnnlp.verbose.config(0)

    pool = mp.Pool(processes=1)
    check = pool.apply_async(_device_count)
    device_count = check.get()

    scores, processes = [], []
    pool = mp.Pool(processes=n_paral)

    if args.n_gpu > 0:
        assert n_paral * args.n_gpu <= device_count, "Not enough GPU devices."

    def get_result(t):
        for i, p in enumerate(processes):
            result = p.get()
            scores.append(result)
            print()
            print(params_search[t + 1 -len(processes) + i])
            ptable = utils.display_prfacc(args.eval_metric, verbose=0)
            ptable.row(dict(result, **{"iter": t + 2 - len(processes) + i}))

    params_search = list(ParameterGrid(params_search))
    for t, params in enumerate(params_search):
        run_params['params_set'] = params
        if args.n_gpu > 0:
            run_params['device_id'] = (t % n_paral) * args.n_gpu
        else:
            run_params['device_id'] = -1

        processes.append(pool.apply_async(
            _pack_run,
            args=(exec_class, run_func),
            kwds=run_params.copy()
        ))

        if (t + 1) % n_paral == 0:
            get_result(t)
            processes.clear()

    get_result(t)

    max_scores = utils.maximum_prfacc(*scores, eval_metric=args.eval_metric)
    print()
    ptable = utils.display_prfacc(args.eval_metric, verbose=0)
    ptable.row(dict(max_scores, **{"iter": 'MAX'}))

    dnnnlp.verbose.config(2)
    return

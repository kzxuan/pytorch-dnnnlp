#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some utilities for deep neural network.
Last update: KzXuan, 2019.10.29
"""
import sys
import dnnnlp
import warnings
import numpy as np
from sklearn.metrics import classification_report
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")


def sysprint(_str):
    """Print without '\n'.

    Args:
        _str [str]: string to output
    """
    sys.stdout.write(_str)
    sys.stdout.flush()


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


def len_to_mask(seq_len, max_seq_len=None):
    """Convert seq_len to mask matrix.

    Args:
        seq_len [tensor]: sequence length vector (batch_size,)
        max_seq_len [int]: max sequence length

    Returns:
        mask [tensor]: mask matrix (batch_size * max_seq_len)
    """
    if isinstance(seq_len, np.ndarray):
        if max_seq_len is None:
            max_seq_len = seq_len.max()
        query = np.arange(0, max_seq_len)
        mask = (query < seq_len.reshape(-1, 1)).astype(int)
    else:
        import torch
        if max_seq_len is None:
            max_seq_len = seq_len.max()
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
        seq_len = mask.sum(dim=1).int()
    return seq_len


def _form_digits(evals, ndigits):
    """Form digits in dict.

    Args:
        evals [dict]: dict of all the evaluation metrics
        ndigits [int]: decimal number of float
    """
    for key, value in evals.items():
        if isinstance(value, float):
            evals[key] = round(float(value), ndigits)
        if isinstance(value, dict):
            _form_digits(value, ndigits)


def prfacc(y_true, y_pred, mask=None, one_hot=False, ndigits=4, tabular=False):
    """Evaluation of true label and prediction.

    Args:
        y_true [np.array/list/torch.Tensor]: true label size of (n_sample, *) / (n_sample, *, n_class)
        y_pred [np.array/list/torch.Tensor]: predict label size of (n_sample, *) / (n_sample, *, n_class)
        mask [np.array/list/torch.Tensor]: mask matrix size of (n_sample, *)
        one_hot [bool]: True for (n_sample, *, n_class) input
        ndigits [int]: decimal number of float
        tabular [bool]: return a table

    Returns:
        evals [dict/str]: dict of all the evaluation metrics or report table
    """
    y_true = np.array(y_true) if isinstance(y_true, list) else y_true
    y_pred = np.array(y_pred) if isinstance(y_pred, list) else y_pred
    mask = np.array(mask) if isinstance(mask, list) else mask

    if not isinstance(y_true, np.ndarray):
        try:
            import torch
            y_true = y_true.cpu().data.numpy() if isinstance(y_true, torch.Tensor) else y_true
            y_pred = y_pred.cpu().data.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
            mask = mask.cpu().data.numpy() if isinstance(mask, torch.Tensor) else mask
        except:
            TypeError("Type error of the input matrices.")

    assert y_true.ndim == y_pred.ndim, "Dimension match error."
    if one_hot:
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

    if mask is not None:
        assert y_true.shape == y_pred.shape == mask.shape, "Dimension error."
        mask_ind = np.where(mask == 1)
        y_true, y_pred = y_true[mask_ind], y_pred[mask_ind]
    else:
        y_true, y_pred = y_true.flatten(), y_pred.flatten()

    n_class = max(np.max(y_true), np.max(y_pred)) + 1
    names = ['class{}'.format(i) for i in range(n_class)]

    evals = classification_report(
        y_true, y_pred, digits=ndigits, target_names=names, output_dict=not tabular
    )

    if not tabular:
        evals['macro'] = evals.pop("macro avg")
        evals['weighted'] = evals.pop("weighted avg")
        p, r = evals['macro']['precision'], evals['macro']['recall']
        if p + r != 0:
            evals['macro']['f1-score'] = 2 * p * r / (p + r)
        _form_digits(evals, ndigits)

    return evals


def average_prfacc(*evals, ndigits=4):
    """Average for multiple evaluations.

    Args:
        evals [tuple]: several evals without limitation
        ndigits [int]: decimal number of float

    Returns:
        avg [dict]: dict of the average values of all the evaluation metrics
    """
    avg = {}.fromkeys(evals[0].keys())
    for key in avg:
        values = [e[key] for e in evals]
        if isinstance(values[0], dict):
            avg[key] = average_prfacc(*values)
        else:
            avg[key] = round(sum(values) / len(values), ndigits)

    return avg


def maximum_prfacc(*evals, eval_metric='accuracy'):
    """Get maximum for multiple evaluations.

    Args:
        evals [tuple]: several evals without limitation
        eval_metric [str]: evaluation standard for comparsion

    Returns:
        max_eval [dict]: one eval with the maximum score
    """
    assert eval_metric in evals[0].keys(), ValueError("Value error of 'eval_metric'.")

    if eval_metric == 'accuracy':
        values = [e[eval_metric] for e in evals]
    else:
        values = [e[eval_metric]['f1-score'] for e in evals]
    index = values.index(max(values))
    max_eval = evals[index]

    return max_eval


class display_prfacc(object):
    """Display evaluations line by line.
    """
    def __init__(self, *eval_metrics, sep='|', verbose=2):
        """Initilize and print head.

        Args:
            eval_metrics [str]: several wanted evaluation metrics
            sep [str]: separate mark like ' '/'|'/'*'
            verbose [int]: verbose level
        """
        self.verbose = verbose
        if not dnnnlp.verbose.check(self.verbose):
            return
        eval_metrics = list(eval_metrics)
        for i, em in enumerate(eval_metrics):
            if em[:5] not in ['accur', 'macro', 'micro', 'class']:
                raise ValueError("Value error of the 'eval_metric'.")
            if em in ['accuracy', 'micro']:
                eval_metrics[i] = 'macro'
        self.eval_metrics = list(set(eval_metrics))
        self.eval_metrics.sort(key=eval_metrics.index)

        self.col = ["iter", "loss", "acc", *self.eval_metrics]
        self.width = [4, 6, 6] + [22] * len(self.eval_metrics)
        self.sep = ' ' + sep + ' '

        for i in range(len(self.col)):
            sysprint(self.sep)
            sysprint("{:^{}}".format(self.col[i], self.width[i]))
        print(self.sep)

    def line(self):
        """Print a line.
        """
        if not dnnnlp.verbose.check(self.verbose):
            return
        for i in range(len(self.col)):
            sysprint(self.sep)
            sysprint("-" * self.width[i])
        print(self.sep)

    def row(self, evals):
        """Process and print a row.
        Atgs:
            evals [dict]: dict of all the evaluation metrics
        """
        if not dnnnlp.verbose.check(self.verbose):
            return
        sysprint(self.sep)
        sysprint("{:^4}".format(evals.get('iter', '-')))
        sysprint(self.sep)
        if 'loss' in evals:
            sysprint("{:^.4f}".format(evals['loss'])[:6])
        else:
            sysprint("{:^6}".format('-'))
        sysprint(self.sep)
        sysprint("{:^.4f}".format(evals['accuracy'])[:6])
        for em in self.eval_metrics:
            sysprint(self.sep)
            value = evals.get(em, '-')
            if value != '-':
                value = "{:.4f}".format(value['precision'])[:6] + "  " +\
                        "{:.4f}".format(value['recall'])[:6] + "  " +\
                        "{:.4f}".format(value['f1-score'])[:6]
            sysprint("{:^22}".format(value))
        print(self.sep)

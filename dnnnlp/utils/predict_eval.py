#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict data analysis
Last update: KzXuan, 2019.08.11
"""
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


def _f1(p, r):
    """Calculate f1 score.

    Args:
        p [float/np.array]: precision score
        r [float/np.array]: recall score
    """
    return (2. * p * r) / (p + r)


def _form_value_type(evals, ndigits):
    """Convert np.float64 to float, np.ndarry to list.

    Args:
        evals [dict]: dict of all the evaluation metrics
        ndigits [int]: decimal number of float
    """
    for key, value in evals.items():
        if isinstance(value, float):
            evals[key] = round(float(value), ndigits)
        if isinstance(value, np.ndarray):
            evals[key] = list(value)
        if isinstance(value, dict):
            _form_value_type(value, ndigits)


def _prfacc(y_true, y_pred):
    """Analyse prediction and give evaluation results.

    Args:
        y_true [np.array/list]: true label size of (n_sample,)
        y_pred [np.array/list]: predict label size of (n_sample,)

    Returns:
        evals [dict]: dict of all the evaluation metrics
                      keys: Acc, Correct, Ma-R, Ma-P, Ma-F, Mi-R, C0-P, C1-F, ...
    """
    assert y_true.ndim == y_pred.ndim == 1, "Dimension error."

    n_class = max(np.max(y_true), np.max(y_pred)) + 1

    # all class statistic
    real = np.array([np.sum(y_true == i) for i in range(n_class)])
    pred = np.array([np.sum(y_pred == i) for i in range(n_class)])

    correct = y_true[np.argwhere((y_true == y_pred) == True).flatten().astype(int)]
    correct = np.array([np.sum(correct == i) for i in range(n_class)])

    # calculate class prf
    precision = np.nan_to_num(correct / pred, 0)
    recall = np.nan_to_num(correct / real, 0)
    f1 = np.nan_to_num(_f1(precision, recall), 0)

    evals = {
        'n_class': n_class,
        'acc': np.sum(correct) / np.sum(real),
        'correct': correct,
        'real': real,
        'pred': pred
    }
    for i in range(n_class):
        evals['class{}-p'.format(i)] = precision[i]
        evals['class{}-r'.format(i)] = recall[i]
        evals['class{}-f'.format(i)] = f1[i]

    evals['micro-p'], evals['micro-r'] = np.mean(correct) / np.mean(pred), np.mean(correct) / np.mean(real)
    evals['micro-f'] = _f1(evals['micro-p'], evals['micro-r'])
    evals['macro-p'], evals['macro-r'] = np.mean(precision), np.mean(recall)
    evals['macro-f'] = _f1(evals['macro-p'], evals['macro-r'])

    return evals


def prfacc1d(y_true, y_pred, one_hot=False, ndigits=4):
    """Evaluation entrance for 1d label.

    Args:
        y_true [np.array/list]: true label size of (n_sample,) / (n_sample, n_class)
        y_pred [np.array/list]: predict label size of (n_sample,) / (n_sample, n_class)
        one_hot [bool]: True for (n_sample, n_class) input
        ndigits [int]: decimal number of float

    Returns:
        evals [dict]: dict of all the evaluation metrics
    """
    y_true = np.array(y_true) if isinstance(y_true, list) else y_true
    y_pred = np.array(y_pred) if isinstance(y_pred, list) else y_pred

    if one_hot:
        assert y_true.ndim == y_pred.ndim == 2, "Dimension error."
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
    else:
        assert y_true.ndim == y_pred.ndim == 1, "Dimension error."
    assert y_true.shape == y_pred.shape, "Dimension error."

    evals = _prfacc(y_true, y_pred)
    _form_value_type(evals, ndigits)

    return evals


def prfacc2d(y_true, y_pred, mask=None, one_hot=False, ndigits=4):
    """Evaluation entrance for 2d label.

    Args:
        y_true [np.array/list]: true label size of (n_sample, seq_len) / (n_sample, seq_len, n_class)
        y_pred [np.array/list]: predict label size of (n_sample, seq_len) / (n_sample, seq_len, n_class)
        one_hot [bool]: True for (n_sample, seq_len, n_class) input
        ndigits [int]: decimal number of float

    Returns:
        evals [dict]: dict of all the evaluation metrics
    """
    y_true = np.array(y_true) if isinstance(y_true, list) else y_true
    y_pred = np.array(y_pred) if isinstance(y_pred, list) else y_pred
    mask = np.array(mask) if isinstance(mask, list) else mask

    if one_hot:
        assert y_true.ndim == y_pred.ndim == 3, "Dimension error."
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
    else:
        assert y_true.ndim == y_pred.ndim == 2, "Dimension error."

    if mask is not None:
        assert y_true.shape == y_pred.shape == mask.shape, "Dimension error."
        mask_ind = np.where(mask == 1)
        y_true, y_pred = y_true[mask_ind], y_pred[mask_ind]
    else:
        y_true, y_pred = y_true.flatten(), y_pred.flatten()

    evals = _prfacc(y_true, y_pred)
    _form_value_type(evals, ndigits)

    return evals


def tabular(evals, class_name=None):
    """Convert evaluations to table expression.

    Args:
        evals [dict]: dict of all the evaluation metrics
        class_name [list]: name of each class

    Returns:
        table [pd.frame]: a pandas table
    """
    assert 'real' in evals, "Use prfacc1d/prfacc2d to get the evaluation dict first."

    n_class = np.size(evals['real'])
    class_name = class_name if class_name else list(range(n_class))
    class_name = class_name + ["avg/micro", "sum/macro"]

    true_pred = evals['correct'] + [round(np.mean(evals['correct']), 1), sum(evals['correct'])]
    pred_class = evals['pred'] + [round(np.mean(evals['pred']), 1), sum(evals['pred'])]
    true_class = evals['real'] + [round(np.mean(evals['real']), 1), sum(evals['real'])]
    precision = [evals['class{}-p'.format(i)] for i in range(n_class)] + [evals['micro-p'], evals['macro-p']]
    recall = [evals['class{}-r'.format(i)] for i in range(n_class)] + [evals['micro-r'], evals['macro-r']]
    f1 = [evals['class{}-f'.format(i)] for i in range(n_class)] + [evals['micro-f'], evals['macro-f']]
    accuracy = [''] * (n_class + 1) + [str(evals['acc'])]

    _tab = [true_pred, pred_class, true_class, precision, recall, f1, accuracy]
    index = ["correct", "predict", "real", "precision", "recall", "f1-score", "accuracy"]
    table = pd.DataFrame(_tab, index=index, columns=class_name)

    return table


def average(*evals, ndigits=4):
    """Average for multiple evaluations.

    Args:
        evals [tuple]: several evals without limitation
        ndigits [int]: decimal number of float

    Returns:
        aver [dict]: dict of the average values of all the evaluation metrics
    """
    aver = {}.fromkeys(evals[0].keys())
    for key in aver:
        values = [e[key] for e in evals]
        aver[key] = np.mean(values, axis=0)
    _form_value_type(aver, ndigits)

    return aver


def maximum(*evals, eval_metric='acc'):
    """Get maximum for multiple evaluations.

    Args:
        evals [tuple]: several evals without limitation
        eval_metric [str]: evaluation standard for comparsion

    Returns:
        max_eval [dict]: one eval with the maximum score
    """
    if eval_metric not in evals[0].keys():
        eval_metric += '-f'
    assert eval_metric in evals[0].keys(), ValueError("Value error of 'eval_metric'.")

    values = [e[eval_metric] for e in evals]
    index = values.index(max(values))
    max_eval = evals[index]

    return max_eval

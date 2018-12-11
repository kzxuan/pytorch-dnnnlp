#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict data analysis
Last update: KzXuan, 2018.12.11
"""
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


def predict_analysis(true, predict, one_hot=False, class_name=None, simple=False, digits=4):
    """
    Analysis the predict by calculation
    * true [array/list]: true label size of (n_sample,) / (n_sample, n_class)
    * predict [array/list]: predict label size of (n_sample,) / (n_sample, n_class)
    * one_hot [bool]: (n_sample, n_class) input needs True
    * class_name [list]: name of each class
    * simple [bool]: simple result or full
    * digits [int]: mumber of decimal places
    """
    if type(true).__name__ == "list":
        true = np.array(true)
    if type(predict).__name__ == "list":
        predict = np.array(predict)

    if one_hot is True:
        predict = np.argmax(predict, axis=1)
        true = np.argmax(true, axis=1)
    n_class = max(np.max(true) + 1, np.max(predict) + 1)
    class_name = list(range(n_class)) if class_name is None else class_name.copy()

    avg_key = 'macro/total'
    class_name.append(avg_key)
    ind, ana = list(), list()

    ind.append('right')  # index 0
    true_pred = np.zeros((n_class,), dtype=np.int)
    for i in range(len(predict)):
        if predict[i] == true[i]:
            true_pred[int(predict[i])] += 1
    ana.append(true_pred.copy())

    ind.append('predict')  # index 1
    pred_class = np.array([np.sum(predict == i) for i in range(n_class)])
    ana.append(pred_class.copy())

    ind.append('true')  # index 2
    true_class = np.array([np.sum(true == i) for i in range(n_class)])
    ana.append(true_class.copy())

    ind.append('precision')  # index 3
    ana.append(np.round(np.nan_to_num(ana[0] / ana[1], 0), digits))

    ind.append('recall')  # index 4
    ana.append(np.round(np.nan_to_num(ana[0] / ana[2], 0), digits))

    ind.append('f1-score')  # index 5
    ana.append(
        np.round(np.nan_to_num((2. * ana[3] * ana[4]) / (ana[3] + ana[4]), 0), digits)
    )

    for i in range(0, 3):
        ana[i] = np.append(ana[i], int(np.sum(ana[i])))
    for i in range(3, 6):
        ana[i] = np.append(ana[i], round(np.mean(ana[i]), digits))

    ind.append('accuracy')  # index 6
    acc = [''] * n_class
    acc.append(round(ana[0][n_class] / ana[2][n_class], digits))
    ana.append(acc)

    if simple:
        detail = {'Acc': ana[6][-1], 'Correct': true_pred}
        detail.update(**{('C%d-P' % c): ana[3][c] for c in range(n_class)}, **{'Ma-P': ana[3][n_class]})
        detail.update(**{('C%d-R' % c): ana[4][c] for c in range(n_class)}, **{'Ma-R': ana[4][n_class]})
        detail.update(**{('C%d-F' % c): ana[5][c] for c in range(n_class)}, **{'Ma-F': ana[5][n_class]})
        return detail
    else:
        ana_str = [np.array(ele, dtype=str) for ele in ana]
        ana_frame = pd.DataFrame(ana_str, index=ind, columns=class_name)
        return ana_frame


def predict_analysis_met(true, predict, one_hot=False, class_name=None, average='binary', digits=4):
    """
    Use metrics to analysis the predict
    * true [array]: true label size of (n_sample,) / (n_sample, n_class)
    * predict [array]: predict label size of (n_sample,) / (n_sample, n_class)
    * one_hot [bool]: (n_sample, n_class) input needs True
    * class_name [list]: name of each class
    * average [str]: use 'binary'/'macro' for binary or multi-class classification
    * digits [int]: mumber of decimal places
    """
    import sklearn.metrics as met

    if one_hot is True:
        predict = np.argmax(predict, axis=1)
        true = np.argmax(true, axis=1)
    if class_name is None:
        n_class = np.max(true) + 1
        class_name = list(range(n_class))
    else:
        n_class = len(class_name)
    avg_key = 'avg / total'
    class_name.append(avg_key)
    ind = list()
    ana = dict([(key, []) for key in class_name])

    ind.append('right')
    true_pred = np.zeros((n_class,), dtype=np.int)
    for i in range(len(predict)):
        if predict[i] == true[i]:
            true_pred[int(predict[i])] += 1
    for i in range(n_class):
        ana[class_name[i]].append(true_pred[i])
    ana[avg_key].append(np.sum(true_pred))

    ind.append('predict')
    pred_class = np.array([np.sum(predict == i) for i in range(n_class)])
    for i in range(n_class):
        ana[class_name[i]].append(pred_class[i])
    ana[avg_key].append(np.sum(pred_class))

    ind.append('true')
    true_class = np.array([np.sum(true == i) for i in range(n_class)])
    for i in range(n_class):
        ana[class_name[i]].append(true_class[i])
    ana[avg_key].append(np.sum(true_class))

    ind.append('precision')
    for i in range(n_class):
        ana[class_name[i]].append(
            round(met.precision_score(true, predict, labels=[i], average=average), digits)
        )
    ana[avg_key].append(
        round(met.precision_score(true, predict, average=average), digits)
    )

    ind.append('recall')
    for i in range(n_class):
        ana[class_name[i]].append(
            round(met.recall_score(true, predict, labels=[i], average=average), digits)
        )
    ana[avg_key].append(
        round(met.recall_score(true, predict, average=average), digits)
    )

    ind.append('f1-score')
    for i in range(n_class):
        ana[class_name[i]].append(
            round(met.f1_score(true, predict, labels=[i], average=average), digits)
        )
    ana[avg_key].append(
        round(met.f1_score(true, predict, average=average), digits)
    )

    ind.append('accuracy')
    for i in range(n_class):
        ana[class_name[i]].append('')
    ana[avg_key].append(
        round(met.accuracy_score(true, predict), digits)
    )

    ana_frame = pd.DataFrame(ana, index=ind, columns=class_name, dtype=str)
    # print(ana_frame)
    return dict(zip(ind, ana)), ana_frame

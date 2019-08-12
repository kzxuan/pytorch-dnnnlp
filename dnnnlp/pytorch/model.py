#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some common models for deep neural network.
Last update: KzXuan, 2019.08.12
"""
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from . import layer


class CNNModel(nn.Module):
    def __init__(self, args, emb_matrix=None, kernel_widths=[2, 3, 4]):
        """Initilize the model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
            kernel_widths [list]: list of kernel widths for CNN kernel
        """
        super(CNNModel, self).__init__()

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        self.cnn = nn.ModuleList()
        for kw in kernel_widths:
            self.cnn.append(layer.CNNLayer(args.emb_dim, 1, args.n_hidden, kw))

        self.predict = layer.SoftmaxLayer(args.n_hidden * len(kernel_widths), args.n_class)

    def forward(self, inputs, mask=None):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)

        Returns:
            pred [tensor]: predict of the model (batch_size * n_class)
        """
        inputs = self.emb_mat(inputs)
        assert inputs.dim() == 3, ValueError("Dimension error of 'inputs', check args.emb_type & emb_dim.")
        if mask is not None:
            assert inputs.shape[:2] == mask.shape, ValueError("Dimension match error of 'inputs' and 'mask'.")

        outputs = self.drop_out(inputs)
        outputs = torch.cat([c(outputs, mask, out_type='max') for c in self.cnn], -1)

        pred = self.predict(outputs)
        return pred


class RNNModel(nn.Module):
    def __init__(self, args, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, mode='LSTM'):
        """Initilize the model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
        """
        super(RNNModel, self).__init__()

        self.n_hierarchy = n_hierarchy
        self.bi_direction_num = 2 if bi_direction else 1

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        rnn_params = (args.n_hidden, n_layer, args.drop_prob, bi_direction, mode)
        self.rnn = nn.ModuleList([layer.RNNLayer(args.emb_dim, *rnn_params)])
        for _ in range(self.n_hierarchy - 1):
            self.rnn.append(layer.RNNLayer(self.bi_direction_num * args.n_hidden, *rnn_params))

        self.predict = layer.SoftmaxLayer(self.bi_direction_num * args.n_hidden, args.n_class)

    def forward(self, inputs, mask=None):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)

        Returns:
            pred [tensor]: predict of the model (batch_size * n_class)
        """
        inputs = self.emb_mat(inputs)
        assert inputs.shape[:-1] == mask.shape, ValueError("Dimension match error of 'inputs' and 'mask'.")
        _, *max_seq_len, _ = inputs.size()
        max_seq_len = max_seq_len[::-1]
        assert len(max_seq_len) == self.n_hierarchy, ValueError("Hierarchy match error of 'inputs'.")

        outputs = self.drop_out(inputs)
        for hi in range(self.n_hierarchy):
            outputs = outputs.reshape(-1, max_seq_len[hi], outputs.size(-1))
            mask = mask.reshape(-1, max_seq_len[hi])
            outputs = self.rnn[hi](outputs, mask, out_type='last')
            mask = (mask.sum(-1) != 0).int()

        pred = self.predict(outputs)
        return pred

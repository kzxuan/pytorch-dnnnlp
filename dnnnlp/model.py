#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some common models for deep neural network.
Last update: KzXuan, 2019.08.28
"""
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from . import layer


class CNNModel(nn.Module):
    def __init__(self, args, emb_matrix=None, kernel_widths=[2, 3, 4], act_fun=nn.ReLU):
        """Initilize CNN model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
            kernel_widths [list]: list of kernel widths for CNN kernel
            act_fun [torch.nn.modules.activation]: activation function
        """
        super(CNNModel, self).__init__()

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        self.cnn = nn.ModuleList()
        for kw in kernel_widths:
            self.cnn.append(
                layer.CNNLayer(args.emb_dim, 1, args.n_hidden, kw, act_fun, args.drop_prob)
            )

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
        assert inputs.dim() == 3, "Dimension error of 'inputs', check args.emb_type & emb_dim."
        if mask is not None:
            assert inputs.shape[:2] == mask.shape, "Dimension match error of 'inputs' and 'mask'."

        outputs = self.drop_out(inputs)
        outputs = torch.cat([c(outputs, mask, out_type='max') for c in self.cnn], -1)

        pred = self.predict(outputs)
        return pred


class RNNModel(nn.Module):
    def __init__(self, args, emb_matrix=None, n_hierarchy=1, n_layer=1,
                 bi_direction=True, rnn_type='LSTM', use_attention=False):
        """Initilize RNN model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
            n_hierarchy [int]: number of model hierarchy
            n_layer [int]: number of RNN layer in a hierarchy
            bi_direction [bool]: use bi-directional model or not
            rnn_type [str]: choose rnn type with 'tanh'/'LSTM'/'GRU'
            use_attention [bool]: use attention layer
        """
        super(RNNModel, self).__init__()

        self.n_hierarchy = n_hierarchy
        self.bi_direction_num = 2 if bi_direction else 1
        self.use_attention = use_attention

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        rnn_params = (args.n_hidden, n_layer, bi_direction, rnn_type, args.drop_prob)
        self.rnn = nn.ModuleList([layer.RNNLayer(args.emb_dim, *rnn_params)])
        if use_attention:
            self.att = nn.ModuleList(
                [layer.SoftAttentionLayer(self.bi_direction_num * args.n_hidden)]
            )
        for _ in range(self.n_hierarchy - 1):
            self.rnn.append(layer.RNNLayer(self.bi_direction_num * args.n_hidden, *rnn_params))
            if use_attention:
                self.att.append(layer.SoftAttentionLayer(self.bi_direction_num * args.n_hidden))

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
        if mask is not None:
            assert inputs.shape[:-1] == mask.shape, "Dimension match error of 'inputs' and 'mask'."

        _, *max_seq_len, _ = inputs.size()
        max_seq_len = max_seq_len[::-1]
        assert len(max_seq_len) == self.n_hierarchy, "Hierarchy match error of 'inputs'."

        outputs = self.drop_out(inputs)
        for hi in range(self.n_hierarchy):
            outputs = outputs.reshape(-1, max_seq_len[hi], outputs.size(-1))
            if mask is not None:
                mask = mask.reshape(-1, max_seq_len[hi])

            if self.use_attention:
                outputs = self.rnn[hi](outputs, mask, out_type='all')
                outputs = self.att[hi](outputs, mask)
            else:
                outputs = self.rnn[hi](outputs, mask, out_type='last')
            if mask is not None:
                mask = (mask.sum(-1) != 0).int()

        pred = self.predict(outputs)
        return pred


class RNNCRFModel(nn.Module):
    def __init__(self, args, emb_matrix=None, n_layer=1, bi_direction=True, rnn_type='LSTM'):
        """Initilize RNN-CRF model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
            n_layer [int]: number of RNN layer in a hierarchy
            bi_direction [bool]: use bi-directional model or not
            rnn_type [str]: choose rnn type with 'tanh'/'LSTM'/'GRU'
        """
        super(RNNCRFModel, self).__init__()

        self.bi_direction_num = 2 if bi_direction else 1

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        rnn_params = (args.n_hidden, n_layer, bi_direction, rnn_type, args.drop_prob)
        self.rnn = layer.RNNLayer(args.emb_dim, *rnn_params)
        self.linear = nn.Linear(self.bi_direction_num * args.n_hidden, args.n_class)
        self.crf = layer.CRFLayer(args.n_class)

    def forward(self, inputs, mask=None, tags=None):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            tags [tensor]: label matrix (batch_size * max_seq_len)

        Returns:
            loss_or_label [tensor]: neg log likelihood loss of the model
                                   or the predict label pad with -1
        """
        inputs = self.emb_mat(inputs)
        assert inputs.dim() == 3, "Dimension error of 'inputs', check args.emb_type & emb_dim."
        if mask is not None:
            assert inputs.shape[:-1] == mask.shape, "Dimension match error of 'inputs' and 'mask'."

        outputs = self.drop_out(inputs)
        outputs = self.rnn(outputs, mask, out_type='all')
        outputs = self.linear(outputs)
        loss_or_label = self.crf(outputs, mask, tags)

        return loss_or_label


class TransformerModel(nn.Module):
    def __init__(self, args, emb_matrix=None, n_layer=6, n_head=8):
        """Initilize transfomer model data and layer.

        Args:
            args [dict]: all model arguments
            emb_matrix [np.array]: word embedding matrix
            n_layer [int]: number of RNN layer in a hierarchy
            n_head [int]: number of attention heads
        """
        super(TransformerModel, self).__init__()

        self.n_layer = n_layer

        self.emb_mat = layer.EmbeddingLayer(emb_matrix, args.emb_type)
        self.drop_out = nn.Dropout(args.drop_prob)

        self.trans = nn.ModuleList(
            [layer.TransformerLayer(args.emb_dim, n_head) for _  in range(n_layer)]
        )
        self.predict = layer.SoftmaxLayer(args.emb_dim, args.n_class)

    def forward(self, inputs, mask=None):
        """Forward propagation.
        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)

        Returns:
            pred [tensor]: predict of the model (batch_size * n_class)
        """
        inputs = self.emb_mat(inputs)
        assert inputs.dim() == 3, "Dimension error of 'inputs', check args.emb_type & emb_dim."
        if mask is not None:
            assert inputs.shape[:2] == mask.shape, "Dimension match error of 'inputs' and 'mask'."

        outputs = self.drop_out(inputs)
        for li in range(self.n_layer - 1):
            outputs = self.trans[li](outputs, query_mask=mask, out_type='all')
        outputs = self.trans[-1](outputs, query_mask=mask, out_type='first')

        pred = self.predict(outputs)
        return pred

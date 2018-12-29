#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some common models for deep neural network
Ubuntu 16.04 & PyTorch 1.0
Last update: KzXuan, 2018.12.29
"""
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from dnn.pytorch import base, layer


class CNN_model(nn.Module, base.base):
    def __init__(self, emb_matrix, args, kernel_widths):
        """
        Initilize the model data and layer
        * emb_matrix [np.array]: word embedding matrix
        * args [dict]: all model arguments
        * kernel_widths [list]: list of kernel widths for cnn kernel
        """
        nn.Module.__init__(self)
        base.base.__init__(self, args)

        self.embedding_layer(emb_matrix)
        self.drop_out = nn.Dropout(self.drop_prob)
        self.cnn = nn.ModuleList()
        for kw in kernel_widths:
            self.cnn.append(layer.CNN_layer(self.emb_dim, 1, self.n_hidden, kw))
        self.predict = layer.softmax_layer(self.n_hidden * len(kernel_widths), self.n_class)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, layer.CNN_layer):
                m.init_weights()
            if isinstance(m, layer.softmax_layer):
                m.init_weights()

    def forward(self, inputs, seq_len):
        """
        Forward calculation of the model
        * inputs [tensor]: model inputs x
        * seq_len [tensor]: sequence length
        - pred [tensor]: predict of the model
        """
        if self.emb_type is not None:
            inputs = self.emb_mat(inputs.long())
        now_batch_size, max_seq_len, emb_dim = inputs.size()

        outputs = self.drop_out(inputs)
        outputs = torch.reshape(outputs, [-1, max_seq_len, outputs.size(-1)])
        seq_len = torch.reshape(seq_len, [-1])
        outputs = torch.cat([c(outputs, seq_len, out_type='max') for c in self.cnn], -1)

        pred = self.predict(outputs)
        return pred


class RNN_model(nn.Module, base.base):
    def __init__(self, emb_matrix, args, mode='classify'):
        """
        Initilize the model data and layer
        * emb_matrix [np.array]: word embedding matrix
        * args [dict]: all model arguments
        * mode [str]: use 'classify'/'sequence' to get the result
        """
        nn.Module.__init__(self)
        base.base.__init__(self, args)

        self.mode = mode
        self.embedding_layer(emb_matrix)
        self.bi_direction_num = 2 if self.bi_direction else 1

        self.drop_out = nn.Dropout(self.drop_prob)

        rnn_params = (self.n_hidden, self.n_layer, self.drop_prob, self.bi_direction, self.GRU_enable)
        self.rnn = nn.ModuleList([layer.LSTM_layer(self.emb_dim, *rnn_params)])
        self.att = nn.ModuleList([layer.self_attention_layer(self.bi_direction_num * self.n_hidden)])
        for _ in range(self.n_hierarchy - 1):
            self.rnn.append(layer.LSTM_layer(self.bi_direction_num * self.n_hidden, *rnn_params))
            if self.use_attention:
                self.att.append(layer.self_attention_layer(self.bi_direction_num * self.n_hidden))
        self.predict = layer.softmax_layer(self.n_hidden * self.bi_direction_num, self.n_class)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, layer.LSTM_layer):
                m.init_weights()
            if isinstance(m, layer.self_attention_layer):
                m.init_weights()
            if isinstance(m, layer.softmax_layer):
                m.init_weights()

    def forward(self, inputs, *seq_len):
        """
        Forward calculation of the model
        * inputs [tensor]: model inputs x
        * seq_len [tensor]: sequence length
        - pred [tensor]: predict of the model
        """
        if self.emb_type is not None:
            inputs = self.emb_mat(inputs.long())  # batch_size * max_seq_len * emb_dim
        now_batch_size, *max_seq_len, emb_dim = inputs.size()
        max_seq_len = max_seq_len[::-1]
        if len(max_seq_len) != self.n_hierarchy:
            raise ValueError("! Parameter 'seq_len' does not correspond to another parameter 'n_hierarchy'.")

        outputs = self.drop_out(inputs)
        for hi in range(self.n_hierarchy - 1):
            outputs = torch.reshape(outputs, [-1, max_seq_len[hi], outputs.size(-1)])
            now_seq_len = torch.reshape(seq_len[hi], [-1])
            if self.use_attention:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='all')
                outputs = self.att[hi](outputs, now_seq_len)
            else:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='last')

        hi = self.n_hierarchy - 1
        outputs = torch.reshape(outputs, [-1, max_seq_len[hi], outputs.size(-1)])
        now_seq_len = torch.reshape(seq_len[hi], [-1])
        if self.mode == 'classify':
            if self.use_attention:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='all')
                outputs = self.att[hi](outputs, now_seq_len)
            else:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='last')  # batch_size * (2)n_hidden
        elif self.mode == 'sequence':
            outputs = self.rnn[hi](outputs, now_seq_len, out_type='all')  # batch_size * max_seq_len * (2)n_hidden

        pred = self.predict(outputs)
        return pred

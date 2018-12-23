#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep neural networks extend model written by PyTorch
Ubuntu 16.04 & PyTorch 1.0.0
Last update: KzXuan, 2018.12.11
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dnn.pytorch import base, layer


class RNN_diachronic_model(nn.Module, base.base):
    def __init__(self, n_time, args):
        nn.Module.__init__(self)
        base.base.__init__(self, args)

        self.n_time = n_time
        self.bi_direction_num = 2 if self.bi_direction else 1
        out_n_hidden = self.n_hidden * self.bi_direction_num
        self.drop_out = nn.Dropout(self.drop_prob)
        self.extractors = nn.ModuleList()
        self.predictors = nn.ModuleList()
        for _ in range(n_time):
            self.extractors.append(
                layer.LSTM_layer(self.emb_dim, self.n_hidden, self.n_layer, self.drop_prob,
                                 self.bi_direction, self.GRU_enable, self.use_attention)
            )  # index 0 -> (nt-1)
            self.predictors.append(layer.softmax_layer(out_n_hidden, self.n_class))  # index 0 -> (nt-1)
        self.connections = nn.ModuleList()
        self.connections.append(None)
        self.connections.extend(
            [nn.Sequential(
                nn.Linear(2 * out_n_hidden, out_n_hidden, bias=False),
                nn.Sigmoid()
             ) for _ in range(n_time - 1)]
        )  # index 1 -> (nt-1)

    def _set_fix_weight(self, now_time):
        for nt in range(now_time):
            for p in self.extractors[nt].parameters():
            for p in self.extractors[nt].parameters():
                p.requires_grad = False

    def forward(self, inputs, seq_len, now_time):
        self._set_fix_weight(now_time)
        inputs = torch.reshape(inputs, [-1, inputs.size(-2), inputs.size(-1)])
        seq_len = torch.reshape(seq_len, [-1])

        outputs = self.drop_out(inputs)
        extractor_out = []
        for nt in range(now_time + 1):
            extractor_out.append(self.extractors[nt](outputs, seq_len, out_type='last'))

        if now_time == 0:
            pred = self.predictors[0](extractor_out[0])
        else:
            conn_in = torch.cat((extractor_out[1], extractor_out[0]), dim=-1)
            conn_out = self.connections[1](conn_in)
            for nt in range(2, now_time + 1):
                conn_in = torch.cat((extractor_out[nt], conn_out), dim=-1)
                conn_out = self.connections[nt](conn_in)
            pred = self.predictors[now_time](conn_out)
        return pred

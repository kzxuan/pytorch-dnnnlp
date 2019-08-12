#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some common layers for deep neural network.
Last update: KzXuan, 2019.08.12
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_matrix, emb_type='const'):
        """Initilize embedding layer.

        Args:
            emb_matrix [tensor/np.array]: embedding matrix
            emb_type [str]: use None/'const'/'variable' to set the layer
        """
        super(EmbeddingLayer, self).__init__()

        # embedding will not be changed
        if emb_type == 'const':
            self.emb_mat = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix))
        # embedding will be changed
        elif emb_type == 'variable':
            self.emb_mat = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix), freeze=False)
        # no embedding
        elif emb_type is None:
            self.emb_mat = None
        else:
            raise ValueError("Value error of 'emb_type', wants None/'const'/'variable', gets '{}'.".format(emb_type))
        self.emb_type = emb_type

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len)

        Returns:
            After-embedding-inputs or original-inputs (batch_size * max_seq_len * emb_dim)
        """
        if self.emb_type is not None:
            return self.emb_mat(inputs.long())
        else:
            return inputs


class SoftmaxLayer(nn.Module):
    def __init__(self, input_size, output_size):
        """Initilize softmax layer / full connected layer.

        Args:
            input_size [int]: number of input nodes
            output_size [int]: number of output nodes
        """
        super(SoftmaxLayer, self).__init__()

        self.input_size = input_size
        self.connector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LogSoftmax(-1)
        )

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs [tensor]: inputs tensor (batch_size * ... * input_size)

        Returns:
            outputs [tensor]: output tensor (batch_size * ... * output_size)
        """
        assert inputs.size(-1) == self.input_size, ValueError("Input size error of 'inputs'.")
        outputs = self.connector(inputs)
        return outputs


class CNNLayer(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_width, act_fun=nn.ReLU):
        """Initilize CNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            in_channels [int]: number of channels for inputs
            out_channels [int]: number of channels for outputs
            kernel_width [int]: the width on sequence for the first dim of kernel
            act_fun [torch.nn.modules.activation]: activation function
        """
        super(CNNLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width

        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_width, input_size))

        assert callable(act_fun), TypeError("Type error of 'act_fun', use functions like nn.ReLU/nn.Tanh.")
        self.act_fun = act_fun()

    def forward(self, inputs, mask=None, out_type='max'):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * in_channels * max_seq_len * input_size)
                             or (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            out_type [str]: use 'max'/'mean'/'all' to choose

        Returns:
            outputs [tensor]: output tensor (batch_size * out_channels) or (batch_size * left_len * n_hidden)
        """
        # auto extend 3d inputs
        if inputs.dim() == 3:
            inputs = torch.unsqueeze(inputs, 1)
            inputs = inputs.repeat(1, self.in_channels, 1, 1)
        assert inputs.dim() == 4 and inputs.size(1) == self.in_channels, ValueError("Dimension error of 'inputs'.")

        now_batch_size, _, max_seq_len, _ = inputs.size()
        assert max_seq_len >= self.kernel_width, ValueError("Dimension error of 'inputs'.")
        assert out_type in ['max', 'mean', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'max'/'mean'/'all'."
        )

        # calculate the seq_len after convolution
        left_len = max_seq_len - self.kernel_width + 1

        # auto generate full-one mask
        if mask is None:
            mask = torch.ones((now_batch_size, left_len), device=inputs.device)
        assert mask.dim() == 2, ValueError("Dimension error of 'mask'.")
        mask = torch.unsqueeze(mask[:, -left_len:], 1)

        outputs = self.conv(inputs)
        outputs = outputs.reshape(-1, self.out_channels, left_len)

        outputs = self.act_fun(outputs)  # batch_size * out_channels * left_len

        # all modes need to consider mask
        if out_type == 'max':
            outputs = outputs.masked_fill(~mask.bool(), -1e10)
            outputs = F.max_pool1d(outputs, left_len).reshape(-1, self.out_channels)
            isinf = outputs.eq(-1e10)
            outputs = outputs.masked_fill(isinf, 0)
        elif out_type == 'mean':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            lens = torch.sum(mask, dim=-1)
            outputs = torch.sum(outputs, dim=-1) / (lens.float() + 1e-9)
        elif out_type == 'all':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            outputs = outputs.transpose(1, 2)  # batch_size * left_len * out_channels

        return outputs


class RNNLayer(nn.Module):
    def __init__(self, input_size, n_hidden, n_layer, drop_prob=0., bi_direction=True, mode="LSTM"):
        """Initilize RNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_hidden [int]: number of hidden layer nodes
            n_layer [int]: number of hidden layers
            drop_prob [float]: drop out ratio
            bi_direction [bool]: use bi-directional model or not
            mode [str]: use 'tanh'/'LSTM'/'GRU' for core model
        """
        super(RNNLayer, self).__init__()

        mode_model = {'tanh': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        assert mode in mode_model.keys(), ValueError("Value error of 'mode', only accepts 'tanh'/'LSTM'/'GRU'.")

        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bi_direction_num = 2 if bi_direction else 1
        self.mode = mode
        self.rnn = mode_model[mode](
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layer,
            bias=True,
            batch_first=True,
            dropout=drop_prob if n_layer > 1 else 0,
            bidirectional=bi_direction
        )

    def forward(self, inputs, mask=None, out_type='all'):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            out_type [str]: use 'all'/'last' to choose

        Returns:
            outputs [tensor]: the last layer output tensor (batch_size * max_seq_len * (bi_direction * n_hidden))
            h_last [tensor]: the last time step output tensor (batch_size * (bi_direction * n_hidden))
        """
        assert inputs.dim() == 3, ValueError("Dimension error of 'inputs'.")
        assert mask is None or mask.dim() == 2, ValueError("Dimension error of 'mask'.")
        assert out_type in ['all', 'last'], ValueError("Value error of 'out_type', only accepts 'all'/'last'.")

        # convert mask to sequence length
        if mask is not None:
            seq_len = mask.sum(dim=-1).int()
        else:
            seq_len = torch.ones((inputs.size(0),))
        # remove full-zero rows
        nonzero_index = torch.nonzero(seq_len).reshape(-1)
        zero_index = torch.nonzero(seq_len == 0).reshape(-1)
        # get rebuild index
        _, re_index = torch.sort(torch.cat((nonzero_index, zero_index)), descending=False)

        inputs = inputs.index_select(0, nonzero_index)
        seq_len = seq_len.index_select(0, nonzero_index)
        now_batch_size, max_seq_len, _ = inputs.size()

        self.rnn.flatten_parameters()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_len, batch_first=True, enforce_sorted=False)
        if self.mode == 'tanh' or self.mode == 'GRU':
            outputs, h_last = self.rnn(inputs)
        elif self.mode == 'LSTM':
            outputs, (h_last, _) = self.rnn(inputs)

        if out_type == 'all':
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=max_seq_len)
            # pad full-zero rows back and rebuild
            outputs = F.pad(outputs, (0, 0, 0, 0, 0, zero_index.size(0)))
            outputs = outputs.index_select(0, re_index)
            return outputs  # batch_size * max_seq_len * (bi_direction * n_hidden)
        elif out_type == 'last':
            h_last = h_last.reshape(self.n_layer, self.bi_direction_num, now_batch_size, self.n_hidden)
            h_last = h_last[-1].transpose(0, 1).reshape(now_batch_size, self.bi_direction_num * self.n_hidden)
            # pad full-zero rows back and rebuild
            h_last = F.pad(h_last, (0, 0, 0, zero_index.size(0)))
            h_last = h_last.index_select(0, re_index)
            return h_last  # batch_size * (bi_direction * n_hidden)



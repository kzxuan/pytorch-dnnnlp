#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some common layers for deep neural network.
Last update: KzXuan, 2019.08.21
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
            raise "Value error of 'emb_type', wants None/'const'/'variable', gets '{}'.".format(emb_type)
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
        assert inputs.size(-1) == self.input_size, "Dimension error of 'inputs'."
        outputs = self.connector(inputs)
        return outputs


class SoftAttentionLayer(nn.Module):
    def __init__(self, input_size):
        """Initilize soft attention layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
        """
        super(SoftAttentionLayer, self).__init__()

        self.input_size = input_size
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 1, bias=False),
        )

    def forward(self, inputs, mask=None):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)

        Returns:
            outputs [tensor]: output tensor (batch_size * input_size)
        """
        assert inputs.dim() == 3, "Dimension error of 'inputs'."
        assert inputs.size(-1) == self.input_size, "Dimension error of 'inputs'."
        assert mask is None or mask.dim() == 2, "Dimension error of 'mask'."

        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).reshape(now_batch_size, 1, max_seq_len)
        exp = alpha.exp()

        if mask is not None:
            exp = exp * mask.unsqueeze(1).float()

        sum_exp = exp.sum(-1, keepdim=True) + 1e-9
        softmax_exp = exp / sum_exp.expand_as(exp).reshape(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(softmax_exp, inputs).squeeze()
        return outputs


class CNNLayer(nn.Module):
    def __init__(self, input_size, in_channels, out_channels,
                 kernel_width, act_fun=nn.ReLU, drop_prob=0.1):
        """Initilize CNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            in_channels [int]: number of channels for inputs
            out_channels [int]: number of channels for outputs
            kernel_width [int]: the width on sequence for the first dim of kernel
            act_fun [torch.nn.modules.activation]: activation function
            drop_prob [float]: drop out ratio
        """
        super(CNNLayer, self).__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width

        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_width, input_size))
        self.drop_out = nn.Dropout(drop_prob)

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
            inputs = inputs.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
        assert inputs.dim() == 4 and inputs.size(1) == self.in_channels, "Dimension error of 'inputs'."
        assert inputs.size(-1) == self.input_size, "Dimension error of 'inputs'."

        now_batch_size, _, max_seq_len, _ = inputs.size()
        assert max_seq_len >= self.kernel_width, "Dimension error of 'inputs'."
        assert out_type in ['max', 'mean', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'max'/'mean'/'all'."
        )

        # calculate the seq_len after convolution
        left_len = max_seq_len - self.kernel_width + 1

        # auto generate full-one mask
        if mask is None:
            mask = torch.ones((now_batch_size, left_len), device=inputs.device)
        assert mask.dim() == 2, "Dimension error of 'mask'."
        mask = mask[:, -left_len:].unsqueeze(1)

        outputs = self.conv(inputs)
        outputs = self.drop_out(outputs)
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
    def __init__(self, input_size, n_hidden, n_layer=1, bi_direction=True,
                 rnn_type='LSTM', drop_prob=0.1):
        """Initilize RNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_hidden [int]: number of hidden layer nodes
            n_layer [int]: number of hidden layers
            bi_direction [bool]: use bi-directional model or not
            rnn_type [str]: choose rnn type with 'tanh'/'LSTM'/'GRU'
            drop_prob [float]: drop out ratio
        """
        super(RNNLayer, self).__init__()

        models = {'tanh': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        assert rnn_type in models.keys(), ValueError("Value error of 'mode', only accepts 'tanh'/'LSTM'/'GRU'.")

        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bi_direction_num = 2 if bi_direction else 1
        self.rnn_type = rnn_type
        self.rnn = models[rnn_type](
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layer,
            bias=True,
            batch_first=True,
            dropout=drop_prob if n_layer > 1 else 0,
            bidirectional=bi_direction
        )

    def forward(self, inputs, mask=None, out_type='last'):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            out_type [str]: use 'all'/'last' to choose

        Returns:
            h_last [tensor]: the last time step output tensor (batch_size * (bi_direction * n_hidden))
            outputs [tensor]: the last layer output tensor (batch_size * max_seq_len * (bi_direction * n_hidden))
        """
        assert inputs.dim() == 3, "Dimension error of 'inputs'."
        assert inputs.size(-1) == self.input_size, "Dimension error of 'inputs'."
        assert mask is None or mask.dim() == 2, "Dimension error of 'mask'."
        assert out_type in ['last', 'all'], ValueError("Value error of 'out_type', only accepts 'last'/'all'.")

        # convert mask to sequence length
        if mask is not None:
            seq_len = mask.sum(dim=-1).int()
        else:
            seq_len = torch.ones((inputs.size(0),))
        # remove full-zero rows
        nonzero_index = seq_len.nonzero().reshape(-1)
        zero_index = (seq_len == 0).nonzero().reshape(-1)
        # get rebuild index
        _, re_index = torch.cat((nonzero_index, zero_index)).sort(descending=False)

        inputs = inputs.index_select(0, nonzero_index)
        seq_len = seq_len.index_select(0, nonzero_index)
        now_batch_size, max_seq_len, _ = inputs.size()

        self.rnn.flatten_parameters()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_len, batch_first=True, enforce_sorted=False)
        if self.rnn_type == 'tanh' or self.rnn_type == 'GRU':
            outputs, h_last = self.rnn(inputs)
        elif self.rnn_type == 'LSTM':
            outputs, (h_last, _) = self.rnn(inputs)

        if out_type == 'last':
            h_last = h_last.reshape(self.n_layer, self.bi_direction_num, now_batch_size, self.n_hidden)
            h_last = h_last[-1].transpose(0, 1).reshape(now_batch_size, self.bi_direction_num * self.n_hidden)
            # pad full-zero rows back and rebuild
            h_last = F.pad(h_last, (0, 0, 0, zero_index.size(0)))
            h_last = h_last.index_select(0, re_index)
            return h_last  # batch_size * (bi_direction * n_hidden)
        elif out_type == 'all':
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=max_seq_len)
            # pad full-zero rows back and rebuild
            outputs = F.pad(outputs, (0, 0, 0, 0, 0, zero_index.size(0)))
            outputs = outputs.index_select(0, re_index)
            return outputs  # batch_size * max_seq_len * (bi_direction * n_hidden)


class CRFLayer(nn.Module):
    def __init__(self):
        super(CRFLayer, self).__init__()


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_size, n_head=8, drop_prob=0.1):
        """Initilize multi-head attention layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_head [int]: number of attention heads
            drop_prob [float]: drop out ratio
        """
        super(MultiheadAttentionLayer, self).__init__()

        self.input_size = input_size
        self.attention = nn.MultiheadAttention(input_size, n_head, drop_prob)

    def forward(self, query, key=None, value=None, query_mask=None, key_mask=None):
        """Forward propagation.

        Args:
            query [tensor]: query tensor (batch_size * max_seq_len_query * input_size)
            key [tensor]: key tensor (batch_size * max_seq_len_key * input_size)
            value [tensor]: value tensor (batch_size * max_seq_len_key * input_size)
            query_mask [tensor]: query mask matrix (batch_size * max_seq_len_query)
            key_mask [tensor]: key mask matrix (batch_size * max_seq_len_key)

        Returns:
            outputs [tensor]: output tensor (batch_size * max_seq_len_query * input_size)
        """
        assert query.dim() == 3, "Dimension error of 'query'."
        assert query.size(-1) == self.input_size, "Dimension error of 'query'."
        # set key = query
        if key is None:
            key, key_mask = query, query_mask
        assert key.dim() == 3, "Dimension error of 'key'."
        assert key.size(-1) == self.input_size, "Dimension error of 'key'."
        # set value = key
        value = key if value is None else value
        assert value.dim() == 3, "Dimension error of 'value'."
        assert value.size(-1) == self.input_size, "Dimension error of 'value'."

        assert query.size(0) == key.size(0) == value.size(0), "Dimension match error."
        assert key.size(1) == value.size(1), "Dimension match error of 'key' and 'value'."
        assert query.size(2) == key.size(2) == value.size(2), "Dimension match error."

        if query_mask is not None:
            assert query_mask.dim() == 2, "Dimension error of 'query_mask'."
            assert query_mask.shape == query.shape[:2], "Dimension match error of 'query' and 'query_mask'."

        # auto generate full-one mask
        if key_mask is None:
            key_mask = torch.ones(key.shape[:2], device=query.device)
        assert key_mask.dim() == 2, "Dimension error of 'key_mask'."
        assert key_mask.shape == key.shape[:2], "Dimension match error of 'key' and 'key_mask'."

        # transpose dimension batch_size and max_seq_len
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        outputs, _ = self.attention(query, key, value, key_padding_mask=~key_mask.bool())
        # transpose back
        outputs = outputs.transpose(0, 1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(2)
            outputs = outputs.masked_fill(~query_mask.bool(), 0)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, input_size, n_head=8, feed_dim=None, drop_prob=0.1):
        """Initilize transformer layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_head [int]: number of attention heads
            feed_dim [int]: hidden matrix dimension
            drop_prob [float]: drop out ratio
        """
        super(TransformerLayer, self).__init__()

        self.input_size = input_size
        self.feed_dim = 4 * self.input_size if feed_dim is None else feed_dim

        self.attention = MultiheadAttentionLayer(input_size, n_head, drop_prob)
        self.drop_out_1 = nn.Dropout(drop_prob)
        self.norm_1 = nn.LayerNorm(input_size)
        self.linear_1 = nn.Linear(input_size, self.feed_dim)
        self.drop_out_2 = nn.Dropout(drop_prob)
        self.linear_2 = nn.Linear(self.feed_dim, input_size)
        self.drop_out_3 = nn.Dropout(drop_prob)
        self.norm_2 = nn.LayerNorm(input_size)

    def forward(self, query, key=None, value=None, query_mask=None,
                key_mask=None, out_type='first'):
        """Forward propagation.

        Args:
            query [tensor]: query tensor (batch_size * max_seq_len_query * input_size)
            key [tensor]: key tensor (batch_size * max_seq_len_key * input_size)
            value [tensor]: value tensor (batch_size * max_seq_len_key * input_size)
            query_mask [tensor]: query mask matrix (batch_size * max_seq_len_query)
            key_mask [tensor]: key mask matrix (batch_size * max_seq_len_key)
            out_type [str]: use 'first'/'all' to choose

        Returns:
            outputs [tensor]: output tensor (batch_size * input_size)
                              or (batch_size * max_seq_len_query * input_size)
        """
        assert query.dim() == 3, "Dimension error of 'query'."
        assert query.size(-1) == self.input_size, "Dimension error of 'query'."
        assert out_type in ['first', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'first'/'all'."
        )

        outputs = self.attention(query, key, value, query_mask, key_mask)
        # residual connection
        outputs = query + self.drop_out_1(outputs)
        outputs = self.norm_1(outputs)

        temp = self.linear_2(self.drop_out_2(F.relu(self.linear_1(outputs))))
        # residual connection
        outputs = outputs + self.drop_out_3(temp)
        outputs = self.norm_2(outputs)

        if query_mask is not None:
            query_mask = query_mask.unsqueeze(2)
            outputs = outputs.masked_fill(~query_mask.bool(), 0)

        if out_type == 'first':
            outputs = outputs[:, 0, :]
        return outputs

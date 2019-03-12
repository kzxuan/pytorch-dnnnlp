#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some extend models written by contributor
Ubuntu 16.04 & PyTorch 1.0.0
Last update: KzXuan, 2019.03.12
"""
import torch
import numpy as np
import torch.nn as nn
import easy_function as ef
import torch.nn.functional as F
from copy import deepcopy
from step_print import table_print
from dnn.pytorch import base, layer, exec
from predict_analysis import predict_analysis


class RNN_diachronic_model(nn.Module, base.base):
    def __init__(self, emb_matrix, args, n_time):
        nn.Module.__init__(self)
        base.base.__init__(self, args)

        self.n_time = n_time
        self.bi_direction_num = 2 if self.bi_direction else 1
        out_n_hidden = self.n_hidden * self.bi_direction_num
        # out_n_hidden = self.n_hidden * 3
        self.drop_out = nn.Dropout(self.drop_prob)
        self.embedding_layer(emb_matrix)

        self.extractors = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.predictors = nn.ModuleList()
        for _ in range(n_time):
            self.extractors.append(
                # layer.LSTM_layer(self.emb_dim, self.n_hidden, self.n_layer, self.drop_prob,
                #                  self.bi_direction, self.GRU_enable)
                # layer.CNN_layer(self.emb_dim, 1, self.n_hidden * 2, 1)
                nn.ModuleList([layer.CNN_layer(self.emb_dim, 1, self.n_hidden, kw) for kw in range(1, 3)])
                # layer.softmax_layer(self.emb_dim, self.n_hidden * 2)
            )  # index 0 -> (nt-1)
            self.attentions.append(layer.self_attention_layer(out_n_hidden))
            self.predictors.append(layer.softmax_layer(out_n_hidden, self.n_class))  # index 0 -> (nt-1)
        self.connections = nn.ModuleList()
        self.connections.append(None)
        for _ in range(n_time - 1):
            self.connections.append(
                nn.Sequential(
                    nn.Linear(2 * out_n_hidden, out_n_hidden, bias=False),
                    nn.Sigmoid()
                )
                # layer.self_attention_layer(out_n_hidden)
            )  # index 1 -> (nt-1)

    def _set_fix_weight(self, now_time):
        for nt in range(now_time):
            for p in self.extractors[nt].parameters():
                p.requires_grad = False

    def reset_fix_weight(self):
        for nt in range(self.n_time):
            for p in self.extractors[nt].parameters():
                p.requires_grad = True

    def forward(self, inputs, seq_len, now_time):
        self._set_fix_weight(now_time)

        inputs = self.emb_mat(inputs.long())
        outputs = self.drop_out(inputs)
        extractor_out = []
        for nt in range(now_time + 1):
            # extractor_out.append(self.extractors[nt](outputs[:, nt], seq_len[:, nt], out_type='last'))
            # nt_out = self.extractors[nt](outputs[:, nt], seq_len[:, nt], out_type='all')
            # nt_out = self.extractors[nt](outputs[:, nt])

            # extractor_out.append(self.extractors[nt](outputs[:, nt], seq_len[:, nt], out_type='max'))
            extractor_out.append(torch.cat([c(outputs[:, nt], seq_len[:, nt]) for c in self.extractors[nt]], -1))
            # extractor_out.append(self.attentions[nt](nt_out, seq_len[:, nt]))
        if now_time == 0:
            pred = self.predictors[0](extractor_out[0])
        else:
            conn_in = torch.cat((extractor_out[1], extractor_out[0]), dim=-1)
            # conn_in = torch.stack((extractor_out[1], extractor_out[0]), dim=1)
            conn_out = self.connections[1](conn_in)
            for nt in range(2, now_time + 1):
                conn_in = torch.cat((extractor_out[nt], conn_out), dim=-1)
                # conn_in = torch.stack((extractor_out[nt], conn_out), dim=1)
                conn_out = self.connections[nt](conn_in)
            pred = self.predictors[now_time](conn_out)
        return pred


class RNN_diachronic_classify(exec.exec):
    def __init__(self, data_dict, emb_matrix, args, n_time):
        self.data_dict = data_dict
        self.n_time = n_time
        base.base.__init__(self, args)

        self.model = RNN_diachronic_model(emb_matrix, args, n_time)
        if self.cuda_enable:
            self.model.cuda()
        self.model_init = deepcopy(self.model.state_dict())

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        self._init_display()

    def _init_display(self):
        self.prf = self.score_standard.split('-')[0] if self.score_standard != 'Acc' else 'Ma'
        self.col = ["Nt", "Step", "Loss", "Ma-P", "Ma-R", "Ma-F", "Acc", "Correct"]
        max_width = np.reshape(self.data_dict['y'], [-1, self.n_class]).shape[0]
        data_scale = (len(str(max_width)) + 1) * self.n_class + 1
        self.width = [4, 4, 6, 6, 6, 6, 6, data_scale]

    def _run_train(self, train_loader, **model_params):
        self.model.train()
        losses = 0
        for step, (x, y, lq) in enumerate(train_loader):
            if self.cuda_enable:
                x, y, lq = x.cuda(), y.cuda(), lq.cuda()
            pred = self.model(x, lq, **model_params)

            nt = model_params['now_time']
            loss = - torch.sum(y.float() * torch.log(pred)) / torch.sum(lq[:, nt]).float()
            losses += loss.cpu().data.numpy()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        losses = losses / (step + 1)
        return losses

    def _run(self, train_loader, test_loader, verbose):
        self.model.reset_fix_weight()
        if verbose > 1:
            ptable = table_print(self.col, self.width, sep="vertical")
        if verbose == 1:
            per = percent("* Run model", self.iter_times)
        best_param = self.model_init
        best_for_nt = []

        for nt in range(self.n_time):
            best_score = -1
            self.model.load_state_dict(best_param)
            for it in range(1, self.iter_times + 1):
                loss = self._run_train(train_loader, now_time=nt)
                pred, ty = self._run_test(test_loader, now_time=nt)
                result = predict_analysis(ty, pred, one_hot=True, simple=True, get_prf='Ma')
                if it % self.display_step == 0 and verbose > 1:
                    ptable.print_row(dict(result, **{"Nt": nt, "Step": it, "Loss": loss}))
                if verbose == 1:
                    per.change()

                if result[self.score_standard] > best_score:
                    best_score = result[self.score_standard]
                    best_param = deepcopy(self.model.state_dict())
            best_for_nt.append(best_score)
        return best_for_nt

    def cross_validation(self, fold=10, verbose=2):
        fold_results = []
        for count, train, test in self.mod_fold(self.data_dict['x'].shape[0], fold=fold):
            train_loader = self.create_data_loader(
                torch.tensor(self.data_dict['x'][train], device=self.device),
                torch.tensor(self.data_dict['y'][train], device=self.device),
                torch.tensor(self.data_dict['len'][train], device=self.device),
            )
            test_loader = self.create_data_loader(
                torch.tensor(self.data_dict['x'][test], device=self.device),
                torch.tensor(self.data_dict['y'][test], device=self.device),
                torch.tensor(self.data_dict['len'][test], device=self.device),
            )

            if verbose > 0:
                _ty = np.reshape(self.data_dict['y'][test], [-1, self.n_class])
                state = np.bincount(np.argmax(ef.remove_zero_rows(_ty)[0], -1))
                print("* Fold {}: {}".format(count, state))

            best_result = self._run(train_loader, test_loader, verbose)
            fold_results.append(best_result)
            if verbose > 0:
                print("* Best scores:", " ".join(["{:.4f}".format(s) for s in best_result]))

        fold_results = np.array(fold_results).mean(0)
        print("* Average scores:", fold_results)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


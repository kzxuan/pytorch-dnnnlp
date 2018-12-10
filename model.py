#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep neural networks model written by PyTorch
Ubuntu 16.04 & PyTorch 1.0
Last update: KzXuan, 2018.12.10
Version 1.7.3
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import utils.easy_function as ef
from copy import deepcopy
from utils.step_print import table_print, percent
from utils.predict_analysis import predict_analysis


def default_args(data_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_enable", default=True, type=bool, help="use GPU to speed up")
    parser.add_argument("--GRU_enable", default=True, type=bool, help="use LSTM or GRU")
    parser.add_argument("--bi_direction", default=True, type=bool, help="bi direction choice")
    parser.add_argument("--n_layer", default=1, type=int, help="hidden layer number")
    parser.add_argument("--use_attention", default=True, type=bool, help="use attention or not")

    parser.add_argument("--emb_type", default=None, type=str, help="embedding type")
    if data_dict is not None:
        parser.add_argument("--emb_dim", default=data_dict['x'].shape[-1], type=int, help="embedding dimension")
        parser.add_argument("--n_class", default=data_dict['y'].shape[-1], type=int, help="classify classes number")
        parser.add_argument("--n_hierarchy", default=len(data_dict['len']), type=int, help="RNN hierarchy number")
    else:
        parser.add_argument("--emb_dim", default=300, type=int, help="embedding dimension")
        parser.add_argument("--n_class", default=2, type=int, help="classify classes number")
        parser.add_argument("--n_hierarchy", default=1, type=int, help="RNN hierarchy number")

    parser.add_argument("--n_hidden", default=50, type=int, help="hidden layer nodes")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--l2_reg", default=1e-6, type=float, help="l2 regularization")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--iter_times", default=30, type=int, help="iteration times")
    parser.add_argument("--display_step", default=2, type=int, help="display inteval")
    parser.add_argument("--drop_prob", default=0.1, type=float, help="drop out ratio")
    parser.add_argument("--score_standard", default='Acc', type=str, help="score standard")
    args = parser.parse_args()
    return args


class RNN(object):
    def __init__(self, args):
        """
        Initilize the model data
        * cuda_enable [bool]: use GPU to speed up
        * GRU_enable [bool]: use LSTM or GRU model
        * bi_direction [bool]: use bi-direction model or not
        * n_layer [int]: number of classify layers
        * use_attention [bool]: use attention or not
        * emb_type [str]: use None/'const'/'variable'/'random'
        * emb_dim [int]: embedding dimension
        * n_class [int]: number of object classify classes
        * n_hierarchy [int]: number of RNN hierarchies
        * n_hidden [int]: number of hidden layer nodes
        * learning_rate [float]: learning rate
        * l2_reg [float]: L2 regularization parameter
        * batch_size [int]: train batch size
        * iter_times [int]: iteration times
        * display_step [int]: the interval iterations to display
        * drop_prob [float]: drop out ratio
        * score_standard [str]: use 'P'/'R'/'F'/'Acc'
        """
        self.cuda_enable = args.cuda_enable
        self.GRU_enable = args.GRU_enable
        self.bi_direction = args.bi_direction
        self.n_layer = args.n_layer
        self.use_attention = args.use_attention
        self.emb_type = args.emb_type
        self.emb_dim = args.emb_dim
        self.n_class = args.n_class
        self.n_hierarchy = args.n_hierarchy
        self.n_hidden = args.n_hidden
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        self.iter_times = args.iter_times
        self.display_step = args.display_step
        self.drop_prob = args.drop_prob
        self.score_standard = args.score_standard

    def attributes_from_dict(self, args):
        """
        Set attributes name and value from dict
        * args [dict]: dict including name and value of parameters
        """
        for name, value in args.items():
            setattr(self, name, value)

    def create_data_loader(self, *data):
        """
        Create data loader for pytorch
        * data [tensor]: several tensors with the same shape[0]
        - loader [DataLoader]: torch data generator
        """
        dataset = Data.TensorDataset(*data)
        loader = Data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
        )
        return loader

    def embedding_layer(self, emb_matrix):
        """
        Initialize embedding place
        * emb_matrix [np.array]: embedding matrix with size (num, dim)
        """
        if self.emb_type == 'const':
            self.emb_mat = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix))
        elif self.emb_type == 'variable':
            self.emb_mat = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix))
            self.emb_mat.weight.requires_grad = True
        elif self.emb_type == 'random':
            self.emb_mat = nn.Embedding(emb_matrix.shape[0], emb_matrix.shape[1])

    @staticmethod
    def mod_fold(length, fold=10):
        """
        Use index for fold
        * length [int]: length of data
        * fold [int]: fold in need
        - indexs [list]: [(fold_num, train_index, test_index),]
        """
        indexs = []
        for f in range(fold):
            test = [ind for ind in range(f, length, fold)]
            train = [ind for ind in range(length) if ind not in test]
            indexs.append((f + 1, train, test))
        return indexs

    @staticmethod
    def vote_sequence(predict, ids):
        """
        Vote for predict/label if the sequence is repetitive
        * predict [np.array]: (sample_num, max_seq_len, n_class)
        * ids [list]: ids of each sequence
        - vote_pred [np.array]: predict array
        """
        predict = np.argmax(predict, axis=2)
        id_sort = [series[-1] for series in ids]
        id_predict = {id: [] for id in id_sort}

        for i, series in enumerate(ids):
            for j, id in enumerate(series):
                id_predict[id].append(predict[i][j])
        for id in id_predict.keys():
            id_predict[id] = np.argmax(np.bincount(id_predict[id]))

        vote_pred = np.array([id_predict[id] for id in id_sort])
        return vote_pred

    def average_several_run(self, run, times=5, **run_args):
        """
        Get average result after several running
        * run [function]: model run function which returns a result dict including 'P'/'R'/'F'/'Acc'
        * times [int]: run several times for average
        * run_args [param]: some parameters for run function including 'fold'/'verbose'
        """
        results = {'P': [], 'R': [], 'F': [], 'Acc': []}

        for i in range(times):
            print("* Run round: {}".format(i + 1))
            result = run(**run_args)
            for key, score in result.items():
                results[key].append(score)
            print("*" * 88)
        for key in results:
            results[key] = ef.list_mean(results[key])
        print("* Average score after {} rounds: {}".format(times, ef.format_dict(results)))

    def grid_search(self, run, params_search=None, **run_args):
        """
        * run [function]: model run function which returns a result dict including 'P'/'R'/'F'/'Acc'
        * params_search [dict]: the argument value need to be tried
        * run_args [param]: some parameters for run function including 'fold'/'verbose'
        """
        from sklearn.model_selection import ParameterGrid

        params_search = list(ParameterGrid(params_search))
        results, max_score = {}, -1

        for params in params_search:
            self.attributes_from_dict(params)
            print("* Now params: {}".format(str(params)))
            result = run(**run_args)
            score = result[self.score_standard]
            results[str(params)] = score
            if score > max_score:
                max_score = score
                max_result = result.copy()
                max_score_params = params
            print("- Result {}: {:6.4f}\n".format(self.score_standard, score))
            print("*" * 88)
        print("* Grid search best:\n  {}\n  {}\n".format(
            ef.format_dict(max_score_params, value_sep=': '), ef.format_dict(max_result))
        )
        print("* All results:\n{}".format(ef.format_dict(results, key_sep='\n', value_sep=': ')))


class self_attention_model(nn.Module):
    def __init__(self, n_hidden):
        """
        Self-attention model
        * n_hidden [int]: hidden layer number (equal to 2*n_hidden if bi-direction)
        """
        super(self_attention_model, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, inputs, seq_len):
        """
        Forward calculation of the model
        * inputs [tensor]: input tensor (batch_size * max_seq_len * n_hidden)
        * seq_len [tensor]: sequence length (batch_size,)
        - outputs [tensor]: attention output (batch_size * n_hidden)
        """
        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).contiguous().view(now_batch_size, 1, max_seq_len)
        seq_len = seq_len.type_as(alpha.data)
        query = torch.arange(0, max_seq_len, device=inputs.device).long().unsqueeze(1).float()
        mask = torch.lt(query, seq_len.unsqueeze(0)).float().transpose(0, 1)
        mask = mask.contiguous().view(now_batch_size, 1, max_seq_len)

        exp = torch.exp(alpha) * mask
        sum_exp = exp.sum(-1, True) + 1e-9
        softmax_exp = exp / sum_exp.expand_as(exp).contiguous().view(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(softmax_exp, inputs).squeeze()
        return outputs


class LSTM_model(nn.Module):
    def __init__(self, input_size, n_hidden, n_layer, drop_prob,
                 bi_direction, GRU_enable=False, use_attention=False):
        """
        LSTM layer model
        * input_size [int]: embedding dim or the last dim of the input
        * n_hidden [int]: number of hidden layer nodes
        * n_layer [int]: number of classify layers
        * n_hidden [int]: number of hidden layer nodes
        * drop_prob [float]: drop out ratio
        * bi_direction [bool]: use bi-direction model or not
        * GRU_enable [bool]: use LSTM or GRU model
        * use_attention [bool]: use attention or not
        """
        super(LSTM_model, self).__init__()
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bi_direction_num = 2 if bi_direction else 1
        self.GRU_enable = GRU_enable
        model = nn.GRU if GRU_enable else nn.LSTM
        self.rnn = model(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layer,
            bias=True,
            batch_first=True,
            dropout=drop_prob if n_layer > 1 else 0,
            bidirectional=bi_direction
        )
        if use_attention:
            self.attention = self_attention_model(self.bi_direction_num * n_hidden)

    def forward(self, inputs, seq_len, out_type='all'):
        """
        Forward calculation of the model
        * inputs [tensor]: input tensor (batch_size * max_seq_len * emb_dim)
        * seq_len [tensor]: sequence length (batch_size,)
        * out_type [str]: out_type [str]: use 'last'/'all'/'att' to choose
        - outputs [tensor]: the last layer (batch_size * max_seq_len * (bi_direction*n_hidden))
        - h_last [tensor]: the last time step of the last layer (batch_size * (bi_direction*n_hidden))
        """
        if inputs.dim() != 3 or seq_len.dim() != 1:
            raise ValueError("! Wrong dimemsion of the input parameters.")

        sort_seq_len, sort_index = torch.sort(seq_len, descending=True)  # sort seq_len
        _, unsort_index = torch.sort(sort_index, dim=0, descending=False)  # get back index
        sort_seq_len = torch.index_select(sort_seq_len, 0, torch.nonzero(sort_seq_len).contiguous().view(-1))
        n_pad = sort_index.size(0) - sort_seq_len.size(0)
        inputs = torch.index_select(inputs, 0, sort_index[:sort_seq_len.size(0)])
        now_batch_size, max_seq_len, _ = inputs.size()

        inputs_pack = nn.utils.rnn.pack_padded_sequence(inputs, sort_seq_len, batch_first=True)
        if self.GRU_enable:
            outputs, h_last = self.rnn(inputs_pack)  # h_last: (n_layer*bi_direction) * batch_size * n_hidden
        else:
            outputs, (h_last, _) = self.rnn(inputs_pack)  # h_last: (n_layer*bi_direction) * batch_size * n_hidden
        if out_type == 'all':
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=max_seq_len  # batch_size * seq_len * (2)n_hidden
            )
            outputs = F.pad(outputs, (0, 0, 0, 0, 0, n_pad))
            outputs = torch.index_select(outputs, 0, unsort_index)
            return outputs
        elif out_type == 'att':
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=max_seq_len  # batch_size * seq_len * (2)n_hidden
            )
            outputs = F.pad(outputs, (0, 0, 0, 0, 0, n_pad))
            outputs = torch.index_select(outputs, 0, unsort_index)
            outputs = self.attention(outputs, seq_len)  # batch_size * (2)n_hidden
            return outputs
        elif out_type == 'last':
            h_last = h_last.contiguous().view(
                self.n_layer, self.bi_direction_num, now_batch_size, self.n_hidden
            )  # n_layer * bi_direction * batch_size * n_hidden
            h_last = torch.reshape(
                h_last[-1].transpose(0, 1),
                [now_batch_size, self.bi_direction_num * self.n_hidden]
            )  # batch_size * (bi_direction*n_hidden)
            h_last = F.pad(h_last, (0, 0, 0, n_pad))
            h_last = torch.index_select(h_last, 0, unsort_index)
            return h_last
        else:
            raise ValueError("! Wrong value of parameter 'out-type', accepts 'all'/'last' only.")


class RNN_model(nn.Module, RNN):
    def __init__(self, emb_matrix, args, model='classify'):
        """
        Initilize the model data and layer
        * emb_matrix [np.array]: word embedding matrix
        * args [dict]: all model arguments
        * model [str]: use 'classify'/'sequence' to get the result
        """
        nn.Module.__init__(self)
        RNN.__init__(self, args)

        self.model = model
        self.embedding_layer(emb_matrix)
        self.bi_direction_num = 2 if self.bi_direction else 1

        self.drop_out = nn.Dropout(self.drop_prob)
        self.rnn = nn.ModuleList()
        self.rnn.append(LSTM_model(self.emb_dim, self.n_hidden, self.n_layer, self.drop_prob,
                                   self.bi_direction, self.GRU_enable, self.use_attention))
        self.rnn.extend([LSTM_model(self.bi_direction_num * self.n_hidden, self.n_hidden, self.n_layer,
                                    self.drop_prob, self.bi_direction, self.GRU_enable, self.use_attention)
                         for _ in range(self.n_hierarchy - 1)])
        self.predict = nn.Sequential(
            nn.Linear(self.n_hidden * self.bi_direction_num, self.n_class),
            nn.Softmax(-1)
        )

    def forward(self, inputs, *seq_len):
        """
        Forward calculation of the model
        * inputs [tensor]: model inputs x
        * seq_len [tensor]: sequence length
        - pred [tensor]: predict of the model
        """
        if self.emb_type is not None:
            inputs = self.emb_mat(inputs.long())  # batch_size * max_seq_len * emb_dim
        outputs = self.drop_out(inputs)
        now_batch_size, *max_seq_len, emb_dim = outputs.size()
        max_seq_len = max_seq_len[::-1]
        if len(max_seq_len) != self.n_hierarchy:
            raise ValueError("! Parameter 'seq_len' does not correspond to another parameter 'n_hierarchy'.")

        for hi in range(self.n_hierarchy - 1):
            outputs = torch.reshape(outputs, [-1, max_seq_len[hi], outputs.size(-1)])
            now_seq_len = torch.reshape(seq_len[hi], [-1])
            if self.use_attention:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='att')
            else:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='last')

        hi = self.n_hierarchy - 1
        outputs = torch.reshape(outputs, [-1, max_seq_len[hi], outputs.size(-1)])
        now_seq_len = torch.reshape(seq_len[hi], [-1])
        if self.model == 'classify':
            if self.use_attention:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='att')
            else:
                outputs = self.rnn[hi](outputs, now_seq_len, out_type='last')
        elif self.model == 'sequence':
            outputs = self.rnn[hi](outputs, now_seq_len, out_type='all')

        pred = self.predict(outputs)
        return pred


class RNN_classify(RNN):
    def __init__(self, data_dict, emb_matrix=None, args=None,
                 class_name=None, col=None, width=None):
        """
        Initilize the LSTM classify model
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * emb_matrix [np.array]: word embedding matrix (need emb_type!=None)
        * args [dict]: all model arguments
        * class_name [list]: name of each class
        * col [list]: name of each column for output
        * form [list]: width of each column for output
        """
        self.data_dict = data_dict
        args = default_args(data_dict) if args is None else args
        RNN.__init__(self, args)

        self.model = RNN_model(emb_matrix, args, model='classify')
        if self.cuda_enable:
            self.model.cuda()
        self.model_init = deepcopy(self.model.state_dict())
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

        self.class_name = class_name
        self.col = ["Step", "Loss", "P", "R", "F", "Acc", "Correct"] if col is None else col
        if width is None:
            data_scale = (len(str(self.data_dict['x'].shape[0])) + 1) * self.n_class + 1
            self.width = [4, 6, 6, 6, 6, 6, data_scale]
        else:
            self.width = width

    def _run_train(self, train_loader):
        """
        Run train part
        * train_loader [DataLoader]: train data generator
        - losses [float]: loss of one iteration
        """
        self.model.train()
        losses = 0.0
        for step, (x, y, *lq) in enumerate(train_loader):
            if self.cuda_enable:
                x, y, lq = x.cuda(), y.cuda(), [ele.cuda() for ele in lq]
            pred = self.model(x, *lq)
            loss = - torch.sum(y.float() * torch.log(pred)) / torch.sum(lq[-1]).float()
            losses += loss.cpu().data.numpy()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        losses = losses / (step + 1)
        return losses

    def _run_test(self, test_loader):
        """
        Run test part
        * test_loader [DataLoader]: test data generator
        - preds [np.array]: predicts of the test data
        - tys [np.array]: true label of the test data
        """
        self.model.eval()
        preds, tys = torch.FloatTensor(), torch.LongTensor()
        for step, (tx, ty, *tlq) in enumerate(test_loader):
            if self.cuda_enable:
                tx, ty, tlq = tx.cuda(), ty.cuda(), [ele.cuda() for ele in tlq]
            pred = self.model(tx, *tlq)

            preds = torch.cat((preds, pred.cpu()))
            tys = torch.cat((tys, ty.cpu()))
        return preds.data.numpy(), tys.data.numpy()

    def _run(self, now_data_dict, verbose):
        """
        Model run part
        * now_data_dict [dict]: data dict with torch.tensor in it
        * verbose [int]: visual hierarchy including 0/1/2
        - best_iter [int]: the iteration number of the best result
        - best_ty [np.array] the true label of the best result
        - best_pred [np.array]: the prediction of the best result
        - best_result [dict]: the best result value including 'P'/'R'/'F'/'Acc'
        """
        train_loader = self.create_data_loader(
            now_data_dict['x'], now_data_dict['y'], *now_data_dict['len']
        )
        test_loader = self.create_data_loader(
            now_data_dict['tx'], now_data_dict['ty'], *now_data_dict['tlen']
        )

        self.model.load_state_dict(self.model_init)
        best_score = -1
        if verbose > 1:
            ptable = table_print(self.col, self.width, sep="vertical")
        if verbose == 1:
            per = percent("* Run model", self.iter_times)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader)
            pred, ty = self._run_test(test_loader)
            true_pred, result = predict_analysis(ty, pred, one_hot=True, simple=True)
            if it % self.display_step == 0 and verbose > 1:
                ptable.print_row([it, loss, result['P'], result['R'], result['F'],
                                  result['Acc'], str(true_pred)])
            if verbose == 1:
                per.change()

            if result[self.score_standard] > best_score:
                best_score = result[self.score_standard]
                best_result = result.copy()
                best_iter = it
                best_ty = deepcopy(ty)
                best_pred = deepcopy(pred)
        return best_iter, best_ty, best_pred, best_result

    def train_test(self, verbose=2):
        """
        Run train and test
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        now_data_dict = {
            'x': torch.FloatTensor(self.data_dict['x']),
            'tx': torch.FloatTensor(self.data_dict['tx']),
            'y': torch.LongTensor(self.data_dict['y']),
            'ty': torch.LongTensor(self.data_dict['ty']),
            'len': [torch.IntTensor(ele) for ele in self.data_dict['len']],
            'tlen': [torch.IntTensor(ele) for ele in self.data_dict['tlen']]
        }

        best_iter, best_ty, best_pred, best_result = self._run(now_data_dict, verbose)
        one_hot = True if best_ty.ndim > 1 else False
        if verbose > 1:
            _, ana = predict_analysis(
                best_ty, best_pred, one_hot=one_hot, class_name=self.class_name, simple=False)
            print("- Best test result: Iteration {}\n".format(best_iter), ana)
        elif verbose > 0:
            true_pred, result = predict_analysis(best_ty, best_pred, one_hot=one_hot, simple=True)
            print("- Best result: It {:2d}, {}, Correct {}".format(
                best_iter, ef.format_dict(result), true_pred))
        return best_result

    def train_itself(self, verbose=2):
        """
        Run test by train data
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        self.data_dict.update({
            'tx': self.data_dict['x'].copy(),
            'ty': self.data_dict['y'].copy(),
            'tlen': self.data_dict['len'].copy()
        })
        return self.train_test(verbose)

    def cross_validation(self, fold=10, verbose=2):
        """
        Run cross validation
        * verbose [int]: visual level including 0/1/2
        - best_result [dict]: the best result with key 'P'/'R'/'F'/'Acc'
        """
        p_kf, r_kf, f_kf, acc_kf = [], [], [], []
        for count, train, test in self.mod_fold(self.data_dict['x'].shape[0], fold=fold):
            now_data_dict = {
                'x': torch.FloatTensor(self.data_dict['x'][train]),
                'tx': torch.FloatTensor(self.data_dict['x'][test]),
                'y': torch.LongTensor(self.data_dict['y'][train]),
                'ty': torch.LongTensor(self.data_dict['y'][test]),
                'len': [torch.IntTensor(ele[train]) for ele in self.data_dict['len']],
                'tlen': [torch.IntTensor(ele[test]) for ele in self.data_dict['len']]
            }

            if verbose > 0:
                state = np.argmax(now_data_dict['ty'].numpy(), axis=1).tolist()
                state = [state.count(i) for i in range(self.n_class)]
                print("* Fold {}: {}".format(count, state))
            best_iter, best_ty, best_pred, best_result = self._run(now_data_dict, verbose)
            one_hot = True if best_ty.ndim > 1 else False
            if verbose > 0:
                true_pred, result = predict_analysis(best_ty, best_pred, one_hot=one_hot, simple=True)
                print("- Best result: It {:2d}, {}, Correct {}".format(
                    best_iter, ef.format_dict(result), true_pred))
                print("-" * 88)
            ef.batch_append([p_kf, r_kf, f_kf, acc_kf], [result['P'], result['R'], result['F'], result['Acc']])
        p, r, f, acc = ef.list_mean(p_kf), ef.list_mean(r_kf), ef.list_mean(f_kf), ef.list_mean(acc_kf)
        result = {'P': p, 'R': r, 'F': f, 'Acc': acc}
        if verbose > 0:
            print("* Avg: {}".format(ef.format_dict(result)))
        return result


class RNN_sequence(RNN_classify):
    def __init__(self, data_dict, emb_matrix=None, args=None,
                 class_name=None, col=None, width=None):
        """
        Initilize the LSTM classify model
        * data_dict [dict]: use key like 'x'/'vx'/'ty'/'lq' to store the data
        * emb_matrix [np.array]: word embedding matrix (need emb_type!=None)
        * args [dict]: all model arguments
        * class_name [list]: name of each class
        * col [list]: name of each column for output
        * form [list]: width of each column for output
        """
        self.data_dict = data_dict
        args = default_args(data_dict) if args is None else args
        RNN.__init__(self, args)

        self.model = RNN_model(emb_matrix, args, model='sequence')
        if self.cuda_enable:
            self.model.cuda()
        self.model_init = deepcopy(self.model.state_dict())
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

        self.class_name = class_name
        self.col = ["Step", "Loss", "P", "R", "F", "Acc", "Correct"] if col is None else col
        if width is None:
            data_scale = (len(str(self.data_dict['x'].shape[0])) + 1) * self.n_class + 1
            self.width = [4, 6, 6, 6, 6, 6, data_scale]
        else:
            self.width = width

    def _run(self, now_data_dict, verbose):
        """
        Model run part
        * now_data_dict [dict]: data dict with torch.tensor in it
        * verbose [int]: visual hierarchy including 0/1/2
        - best_iter [int]: the iteration number of the best result
        - best_ty [np.array] the true label of the best result
        - best_pred [np.array]: the prediction of the best result
        - best_result [dict]: the best result value including 'P'/'R'/'F'/'Acc'
        """
        train_loader = self.create_data_loader(
            now_data_dict['x'], now_data_dict['y'], *now_data_dict['len']
        )
        test_loader = self.create_data_loader(
            now_data_dict['tx'], now_data_dict['ty'], *now_data_dict['tlen']
        )

        self.model.load_state_dict(self.model_init)
        best_score = -1
        if verbose > 1:
            ptable = table_print(self.col, self.width, sep="vertical")
        if verbose == 1:
            per = percent("* Run model", self.iter_times)
        for it in range(1, self.iter_times + 1):
            loss = self._run_train(train_loader)
            pred, ty = self._run_test(test_loader)

            vote_pred = self.vote_sequence(pred, self.data_dict['tid'])
            vote_ty = self.vote_sequence(ty, self.data_dict['tid'])

            true_pred, result = predict_analysis(vote_ty, vote_pred, one_hot=False, simple=True)
            if it % self.display_step == 0 and verbose > 1:
                ptable.print_row([it, loss, result['P'], result['R'], result['F'],
                                  result['Acc'], str(true_pred)])
            if verbose == 1:
                per.change()

            if result[self.score_standard] > best_score:
                best_score = result[self.score_standard]
                best_result = result.copy()
                best_iter = it
                best_ty = deepcopy(vote_ty)
                best_pred = deepcopy(vote_pred)
        return best_iter, best_ty, best_pred, best_result

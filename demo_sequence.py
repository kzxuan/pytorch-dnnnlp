#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use lstm sequence model
Last update: KzXuan, 2018.12.08
"""
import numpy as np
import word_vector as wv
import utils.easy_function as ef
from utils.dir_set import dir
from utils.step_print import slash, run_time
from model import default_args, RNN_sequence

sl, rt = slash(), run_time("* PyTorch LSTM sequence model")

w2v = wv.load_word2vec(dir.W2V_GOOGLE, type='txt')
emb_mat = w2v.get_matrix()
print("- Embedding matrix size:", emb_mat.shape)

sl.start("* Load data")
data_dict = {
    'x': np.load(dir.TRAIN + "index(4519,3,30).npy"),
    'y': np.load(dir.TRAIN + "y(4519,3,4).npy"),
    'len': [np.load(dir.TRAIN + "len_sen(4519,3).npy"), np.load(dir.TRAIN + "len_seq(4519,).npy")],
    'id': ef.load_list(dir.TRAIN + "id.txt"),
    'tx': np.load(dir.TEST + "index(1049,3,30).npy"),
    'ty': np.load(dir.TEST + "y(1049,3,4).npy"),
    'tlen': [np.load(dir.TEST + "len_sen(1049,3).npy"), np.load(dir.TEST + "len_seq(1049,).npy")],
    'tid': ef.load_list(dir.TEST + "id.txt"),
}
sl.stop()
ef.print_shape(data_dict)

args = default_args(data_dict)
args.GRU_enable = True
args.use_attention = True
args.emb_type = 'const'
args.emb_dim = w2v.vector_size
args.n_hidden = 50
args.learning_rate = 0.01
args.l2_reg = 0.0
args.batch_size = 64
args.iter_times = 20
args.display_step = 1
args.drop_porb = 0.1

class_name = ['support', 'deny', 'query', 'comment']
nn = RNN_sequence(data_dict, emb_mat, args, class_name)
nn.train_test()

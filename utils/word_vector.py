#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for word2vector operation
Last update: KzXuan, 2018.12.09
"""
import numpy as np
import codecs as cs
from collections import OrderedDict
from step_print import slash, percent

sl = slash()


class word2vector(object):
    """
    A word2vector model
    """
    def __init__(self, vector_size, add_zero=True, zero_key='0000'):
        self.vector_size = vector_size
        self.index = OrderedDict()
        self.vocab = OrderedDict()
        if add_zero:
            self.index[zero_key] = len(self.vocab)
            self.vocab[zero_key] = np.zeros((vector_size,))

    def __getitem__(self, key):
        return self.vocab[key]

    def __setitem__(self, key, value):
        self.index[key] = len(self.vocab)
        self.vocab[key] = value

    def get_matrix(self):
        vector_matrix = []
        for word in self.vocab:
            vector_matrix.append(self.vocab[word])
        return np.array(vector_matrix)

    def update(self, up_w2v):
        if up_w2v.vector_size != self.vector_size:
            raise("! Wrong size.")
        per = percent("* Update word2vector", len(up_w2v.vocab))
        for word in up_w2v.vocab:
            if word not in self.vocab:
                self.index[word] = up_w2v[word]
                self.vocab[word] = up_w2v[word]
            per.change()
        return 1


def load_word2vec(in_dir, type='bin', header=True, binary=True):
    """
    Read word2vector original file
    * type [string]: use 'bin' to load '.bin' file and use 'txt' to load '.txt' file
    * binary [bool]: parameter in load_word2vec_format (need type='bin')
    """
    if type == 'bin':
        sl.start("* Load word2vector")
        from gensim.models.keyedvectors import KeyedVectors
        # logging.basicConfig(format="* %(asctime)s: %(levelname)s: %(message)s",
        #                     level=logging.INFO)
        word_vectors = KeyedVectors.load_word2vec_format(in_dir, binary=binary)
        # word_vectors.save_word2vec_format("./features/w2v.bin", binary=False)
        sl.stop()
    elif type == 'txt':
        with cs.open(in_dir) as fobj:
            lines = fobj.readlines()
        if header is True:
            num_word, emb_dim = [int(ele) for ele in lines[0].split()]
            lines = lines[1:]
        else:
            num_word = len(lines)
            emb_dim = len(lines[0].split(' ')) - 1
        per = percent("* Load word2vector", num_word)
        word_vectors = word2vector(emb_dim, add_zero=False)
        for line in lines:
            word = line.split(' ', 1)[0]
            vector = np.array([float(n) for n in line.split()[-emb_dim:]])
            word_vectors[word] = vector
            per.change()
        word_vectors.vector_size = emb_dim
    else:
        raise("! Wrong type.")
    print("- Word2vector size:", word_vectors.vector_size)
    return word_vectors


def output_word2vec(w2v, out_dir):
    """
    Output word2vec class
    * w2v <word2vector>: word2vector class element
    * out_dir [str]: output file path
    """
    per = percent("* Output word2vec", len(w2v.vocab))
    with cs.open(out_dir, 'w') as fobj:
        fobj.write(str(len(w2v.vocab)) + ' ' + str(w2v.vector_size) + '\n')
        for word, vec in w2v.vocab.items():
            fobj.write(word + ' ' + ' '.join([str(n) for n in vec]) + '\n')
            per.change()
    return 1


def text2vec(text, w2v, model='mean', padding=10):
    """
    Use word2vec mean to represent a tweet text
    * text [list]: the text composed of vocab
    * w2v [word2vector]: word2vector dict
    * model [string]: use 'mean' to get the mean value of word vectors
                      use 'max' to get the max value of each dim
                      use 'index' to get the index of each word
    * padding [int]: complement the vector to a fixed value
                     0 means no padding (need model='index')
    """
    if model == 'mean':
        mean_vec = [w2v[word] for word in text if word in w2v.vocab]
        mean_vec = np.mean(np.array(mean_vec), 0) if len(mean_vec) else np.zeros(w2v.vector_size)
        return mean_vec
    elif model == 'max':
        max_vec = [w2v[word] for word in text if word in w2v.vocab]
        max_vec = np.max(np.array(max_vec), 0) if len(max_vec) else np.zeros(w2v.vector_size)
        return max_vec
    elif model == 'index':
        index_vec = [w2v.index[word] for word in text if word in w2v.vocab]
        if padding != 0 and len(index_vec) > padding:
            raise ValueError("! Padding is too small for the text.")
        length = len(index_vec)
        if padding:
            index_vec += [0] * (padding - len(index_vec))
        index_vec = np.array(index_vec)
        return index_vec, length
    else:
        raise ValueError("! Wrong parameter")


def output_text2vec(text_dict, out_dir):
    """
    Output text embedding vector
    * text_dict [dict]: {text/id: embedding}
    * out_dir [str]: output file path
    """
    per = percent("* Output text2vec", len(text_dict))
    with cs.open(out_dir, 'w', 'utf-8') as fobj:
        for key, vec in text_dict.items():
            fobj.write("{}\t{}\n".format(str(key), str(vec)))
            per.change()


def simplify_w2v(w2v, word_list, out_dir, add_zero=True, rand_not_in=False):
    """
    Simplify the initial word2vector use the useful vocab
    * w2v [word2vector]: the initial word2vector
    * word_list [list]: the list of the useful vocab
    * out_dir [str]: the output dir of simplify word2vector
    * add_zero [bool]: add zero vector or not
    * rand_not_in [bool]: random a vector for word not in w2v
    """
    sim_w2v = word2vector(w2v.vector_size, add_zero=add_zero)

    def add_word(word):
        if len(word.split()) != 1:
            return 0
        if word in w2v.vocab:
            sim_w2v[word] = w2v[word]
        else:
            if rand_not_in:
                sim_w2v[word] = np.random.uniform(-0.01, 0.01, w2v.vector_size)
            if word.lower() in w2v.vocab:
                sim_w2v[word.lower()] = w2v[word.lower()]
            # if word.upper() in w2v.vocab:
            #     sim_w2v[word.upper()] = w2v[word.upper()]
            # if ps.stem(word) in w2v.vocab:
            #     sim_w2v[ps.stem(word)] = w2v[ps.stem(word)]
        return 1

    per = percent("* Simplify word2vector", len(word_list))
    for word in word_list:
        add_word(word)
        per.change()

    output_word2vec(sim_w2v, out_dir)
    return 1

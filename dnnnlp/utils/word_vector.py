#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for word embedding operation
Last update: KzXuan, 2019.07.29
"""
import dnnnlp
import numpy as np
import codecs as cs
from .display_tool import bar, dot
from collections import OrderedDict


class word_vector(object):
    """A word2vector class.
    """
    def __init__(self, vector_size, add_zero=True):
        """Initialize.

        Args:
            vector_size [int]: vector size/dim
            add_zero [bool]: add zero index first
        """
        self.vector_size = vector_size
        self.index = OrderedDict()
        self.vocab = OrderedDict()
        if add_zero:
            self.add("#0#0#0", np.zeros((vector_size,)))

    def add(self, word, vector):
        """Add a word.

        Args:
            word [str]: a word
            vector [np.array]: the corresponding vector
        """
        if word not in self.vocab:
            self.index[word] = len(self.vocab)
            self.vocab[word] = vector

    def __getitem__(self, word):
        assert word in self.vocab, KeyError("Can not find this word.")
        return self.vocab[word]

    def __setitem__(self, word, vector):
        self.add(word, vector)

    def get_matrix(self):
        """Get embedding matrix.

        Returns:
            matrix [np.array]: embedding matrix
        """
        vector_matrix = list(self.vocab.values())
        return np.array(vector_matrix)

    def update(self, w2v):
        """Merge two w2vs.

        Args:
            w2v [word_vector]: extend w2v
        """
        assert w2v.vector_size == self.vector_size, "Vector size error of 'w2v'."
        for word, vector in bar(self.vocab):
            self.add(word, vector)
        return 1


def load_w2v(file, type='txt', header=True, check_zero=True, verbose=1):
    """Load word embedding original file.

    Args:
        file [str]: file path
        type [str]: use 'bin'/'txt' for '.bin'/'.txt' file
        check_zero [bool]: check whether the first line is a zero vector (need type='txt')
        verbose [int]: verbose level

    Returns:
        w2v [word_vector]: word_vector class
    """
    if type == 'bin':
        dot.start("* Load word embedding", verbose=verbose)
        from gensim.models.keyedvectors import KeyedVectors
        w2v = KeyedVectors.load_word2vec_format(file, binary=True)
        dot.stop()
    elif type == 'txt':
        with cs.open(file) as fobj:
            line = fobj.readline().rstrip()
            if header:
                num_word, vector_size = map(int, line.split())
                line = fobj.readline().rstrip()
            else:
                vector_size = len(line.split(' ')) - 1

            if check_zero:
                first_vec = list(map(float, line.split(' ')[1:]))
                add_zero = True if any(first_vec) else False
            else:
                add_zero = False

            w2v = word_vector(vector_size, add_zero)
            if header:
                for _ in bar(num_word, "* Load word embedding", verbose=verbose):
                    line = line.rstrip().split(' ')
                    word, vector = line[0], np.array(line[1:], dtype=float)
                    w2v[word] = vector
                    line = fobj.readline()
            else:
                dot.start("* Load word embedding", verbose=verbose)
                while line:
                    line = line.rstrip().split(' ')
                    word, vector = line[0], np.array(line[1:], dtype=float)
                    w2v[word] = vector
                    line = fobj.readline()
                dot.stop()
    else:
        raise ValueError("Value error of 'type', wants 'txt'/'bin', gets '{}'.".format(type))
    if dnnnlp.verbose.check(verbose):
        print("- Word embedding size:", vector_size)
    return w2v


def save_w2v(w2v, file, verbose=1):
    """Output word embedding.

    Args:
        w2v [word_vector]: word_vector class
        file [str]: save file/path
        verbose [int]: verbose level
    """
    with cs.open(file, 'w') as fout:
        fout.write("{} {}\n".format(len(w2v.vocab), w2v.vector_size))
        for word, vector in bar(w2v.vocab, "* Save word embedding", verbose=verbose):
            fout.write(word + ' ' + ' '.join(map(str, vector)) + '\n')
    return 1


def simplify_w2v(w2v, word_list, out_file=None, add_zero=True, rand_not_in=False, verbose=1):
    """Simplify the word embedding with useful words.

    Args:
        w2v [word_vector]: the initial word_vector
        word_list [list]: the list of the useful vocab
        out_file [str]: the output dir of simplify word_vector
        add_zero [bool]: add zero vector or not
        rand_not_in [bool]: random a vector for word not in w2v
        verbose [int]: verbose level

    Returns:
        sim_w2v [word_vector]: word_vector after simplification
    """
    sim_w2v = word_vector(w2v.vector_size, add_zero=add_zero)

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
        return 1

    for word in bar(word_list, "* Simplify word embedding", verbose=verbose):
        add_word(word)

    if out_file:
        save_w2v(sim_w2v, out_file, verbose)
    return sim_w2v


def text_vector(text, w2v, mode='mean', padding=0):
    """Use word_vector to represent a text.

    Args:
        text [list]: the text composed of vocab
        w2v [word_vector]: word vector class
        mode [string]: use 'mean' to get the mean value of word vectors
                       use 'max' to get the max value of each dim
                       use 'joint' to get all the values
                       use 'index' to get the index of each word
        padding [int]: complement the vector to a fixed value
                       0 means no padding (need model='joint'/'index')

    Returns:
        mean_vec/max_vec/joint_vec/index_vec [np.array]: mean/max/joint/index value
        sen_len [int]: true sentence len of joint_vec/index_vec
    """
    if mode == 'mean':
        mean_vec = [w2v[word] for word in text if word in w2v.vocab]
        mean_vec = np.mean(np.array(mean_vec), 0) if len(mean_vec) else np.zeros(w2v.vector_size)
        return mean_vec
    elif mode == 'max':
        max_vec = [w2v[word] for word in text if word in w2v.vocab]
        max_vec = np.max(np.array(max_vec), 0) if len(max_vec) else np.zeros(w2v.vector_size)
        return max_vec
    elif mode == 'joint':
        joint_vec = [w2v[word] for word in text if word in w2v.vocab]
        sen_len = len(joint_vec)
        if padding and sen_len < padding:
            joint_vec.extend([np.zeros(w2v.vector_size) for _ in range(padding - sen_len)])
        elif padding and sen_len > padding:
            joint_vec = joint_vec[:padding]
            sen_len = padding
        joint_vec = np.array(joint_vec)
        return joint_vec, sen_len
    elif mode == 'index':
        index_vec = [w2v.index[word] for word in text if word in w2v.vocab]
        sen_len = len(index_vec)
        if padding and sen_len < padding:
            index_vec += [0] * (padding - sen_len)
        elif padding and sen_len > padding:
            index_vec = index_vec[:padding]
            sen_len = padding
        index_vec = np.array(index_vec)
        return index_vec, sen_len
    else:
        raise ValueError("Value error of 'mode', wants 'mean'/'max'/'joint'/'index', gets '{}'.".format(mode))


def doc_vector(doc, w2v, mode='mean', padding=0, verbose=2):
    """Use word_vector to represent a document.

    Args:
        doc [list]: the document composed of texts
        w2v [word_vector]: word vector class
        mode [string]: use 'mean' to get the mean value of word vectors
                       use 'max' to get the max value of each dim
                       use 'joint' to get all the values
                       use 'index' to get the index of each word
        padding [int]: complement the vector to a fixed value
                       0 means no padding

    Returns:
        vecs [np.array]: mean/max/joint/index value
        doc_len [int]: true document len
        sen_len [int]: true sentence len of joint_vec/index_vec
    """
    max_sen_len = max([len(text) for text in doc])
    vecs, sen_len = [], []
    if mode in ['mean', 'max']:
        for text in bar(doc, "* Convert document to vector"):
            vecs.append(text_vector(text, w2v, mode, max_sen_len))
        doc_len = len(vecs)
        if padding and len(vecs) < padding:
            vecs += [np.zeros((w2v.vector_size,))] * (padding - len(vecs))
        return np.array(vecs), doc_len
    elif mode in ['joint', 'index']:
        for text in bar(doc, "* Convert document to vector"):
            _v, _l = text_vector(text, w2v, mode, max_sen_len)
            vecs.append(_v), sen_len.append(_l)
        doc_len = len(vecs)
        if padding and len(vecs) < padding:
            if mode == 'joint':
                vecs += [np.zeros((max_sen_len, w2v.vector_size))] * (padding - len(vecs))
                sen_len += [0] * (padding - len(sen_len))
            if mode == 'index':
                vecs += [np.zeros((max_sen_len,))] * (padding - len(vecs))
                sen_len += [0] * (padding - len(sen_len))
        return np.array(vecs), doc_len, np.array(sen_len)
    else:
        raise ValueError("Value error of 'mode', wants 'mean'/'max'/'joint'/'index', gets '{}'.".format(mode))

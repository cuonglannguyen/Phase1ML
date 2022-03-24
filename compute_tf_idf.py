# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:58:48 2022

@author: Admin
"""
import numpy as np
from collections import defaultdict
import re
from os import listdir
from os.path import join, isfile
def get_tf_dif(data_path):
    with open('D:/E/CUONG/machine learning/20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_IDs = dict([(word, index) for index, (word,idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)
    with open(data_path) as f:
        documents = [(int(line.split('<fff')[0]), int(line.split('<fff>')[1]), line.split('<fff>')[2]) for line in f.read().splitlines()]
        data_tf_idf = []
        for document in documents:
            label, doc_id, text = document
            words =[word for word in text.split() if word in idfs]
            word_set = list(set(words))
            max_term_freq = max([words.count(word) for word in word_set])
            words_tfidfs = []
            sum_squares = 0.0
            for word in word_set:
                term_freq = words.count(word)
                tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
                words_tfidfs.append((word_IDs[word], tf_idf_value))
                sum_squares += tf_idf_value ** 2
            words_tfidfs_normalized = [str(index) + ':' +str(tf_idf_value / np.sqrt(sum_squares)) for index, tf_idf_value in words_tfidfs]
            sparse_rep = ' '.join(words_tfidfs_normalized)
            data_tf_idf.append((label, doc_id, sparse_rep))
    with open('D:/E/CUONG/machine learning/20news-bydate/20news-full-tfidf.txt', 'w') as f:
        f.write('\n'.join([str(label) + '<fff>'+str(doc_id) +'<fff>'+ str(sparse_rep) for label, doc_id, sparse_rep in data_tf_idf]))
get_tf_dif('D:/E/CUONG/machine learning/20news-bydate/20news-full-processed.txt')
            
            
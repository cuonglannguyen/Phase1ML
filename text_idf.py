# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:34:05 2022

@author: Admin
"""
import numpy as np
from collections import defaultdict
import re
from os import listdir
from os.path import join, isfile
def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)
    for line in lines:
        feature = line.split('<fff>')
        text = feature[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
    words_idfs = [(word, compute_idf(document_freq, corpus_size)) for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key = lambda word, idf: -idf)
    print('Vocabulary size:{}'.format(len(words_idfs)))
    with open('D:/E/CUONG/machine learning/20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>'+str(idf) for word, idf in words_idfs]))
# def gather_20newsgroup_data():
#     path = 'D:/E/CUONG/machine learning/20news-bydate/'
#     dirs = [path + dir_name + '/' for dir_name in listdir(path) if not isfile(path + dir_name)]
#     train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
#     list_newsgroups = [newsgroup for newsgroup in listdir(train_dir)]
#     list_newsgroups.sort()
#     with open('D:/E/CUONG/machine learning/20news-bydate/stop_words.txt') as f:
#         stop_words = f.read().splitlines()
#     from nltk.stem.porter import PorterStemmer
#     stemmer = PorterStemmer()
#     def collect_data_from(parent_dir, newsgroup_list):
#         data = []
#         for group_id, newsgroup in enumerate(newsgroup_list):
#             label = group_id
#             dir_path = parent_dir + '/' + newsgroup + '/'
#             files = [(filename, dir_path + filename) for filename in listdir(dir_path) if isfile(dir_path + filename)]
#             files.sort()
#             for filename, filepath in files:
#                 with open(filepath) as f:
#                     text = f.read().lower()
#                     words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
#                     content = ' '.join(words)
#                     assert len(content.splitlines()) == 1
#                     data.append(str(label) + '<fff>' + filename + '<fff>' + content)
#         return data
                    
#     train_data = collect_data_from(parent_dir = train_dir, newsgroup_list = list_newsgroups)
#     test_data = collect_data_from(parent_dir = test_dir, newsgroup_list = list_newsgroups)
#     full_data = train_data + test_data
#     with open('D:/E/CUONG/machine learning/20news-bydate/20news-train-processed.txt', 'w') as f:
#         f.write('\n'.join(train_data))
#     with open('D:/E/CUONG/machine learning/20news-bydate/20news-test-processed.txt', 'w') as f:
#         f.write('\n'.join(test_data))
#     with open('D:/E/CUONG/machine learning/20news-bydate/20news-full-processed.txt', 'w') as f:
#         f.write('\n'.join(full_data))
# gather_20newsgroup_data()
generate_vocabulary('D:/E/CUONG/machine learning/20news-bydate/20news-train-processed.txt')

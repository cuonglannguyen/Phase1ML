# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:40:50 2022

@author: Admin
"""

import sklearn
import scipy
from collections import defaultdict
import numpy as np
import random

def load_data(data_path):
     def sparse_to_dense(sparse_r_d, vocab_size):
         r_d = [0.0 for _ in range(vocab_size)]
         indices_tfidfs = sparse_r_d.split()
         for index_tfidf in indices_tfidfs:
             index = int(index_tfidf.split(':')[0])
             tfidf = float(index_tfidf.split(':')[1])
             r_d[index] = tfidf
             return np.array(r_d)
     with open(data_path) as f:
         d_lines = f.read().splitlines()
     with open('D:/E/CUONG/machine learning/20news-bydate/words_idfs.txt') as f:
         vocab_size = len(f.read().splitlines())
     data = []
     label_count = defaultdict(int)
     for data_id, d  in enumerate(d_lines):
         features = d.split('<fff>')
         label, doc_id = int(features[0]), int(features[1])
         label_count[label] += 1
         r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
         data.append([r_d.tolist(), [label],[doc_id]])
     return data, label_count
def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy
'''      
Uncomment to implement clustering with Kmeans
# def clustering_with_Kmeans():
#     data, labels = load_data(data_path = 'D:/E/CUONG/machine learning/20news-bydate/20news-full-tfidf.txt')
#     new_data =[]
#     for t in data:
#         flat_list = [item for sublist in t for item in sublist]
#         new_data.append(flat_list)


#     from sklearn.cluster import KMeans
#     from scipy.sparse import csr_matrix
#     data = np.array(new_data, dtype=float)

#     X = csr_matrix(data)
#     print('========')
#     kmeans = KMeans(n_clusters=20, init = 'random',n_init = 5, tol = 1e-3, random_state = 2018).fit(X)
#     labels = kmeans.labels_
#     print(kmeans)
#     print(labels)
#     print(kmeans.cluster_centers_)

# clustering_with_Kmeans()
           '''
def load_data_for_linear_SVMs(data_path):
    
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
            return np.array(r_d)
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('D:/E/CUONG/machine learning/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())
    data = []
    labels = []
    for data_id, d  in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
        data.append(r_d)
        labels.append(label)
        
    return data, np.array(labels)
def classifying_with_linear_SVMs():
    train_X, train_Y = load_data_for_linear_SVMs('D:/E/CUONG/machine learning/20news-bydate/20news-train-tfidf.txt')
    
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(C = 10.0, tol = 0.001, verbose = True)
    classifier.fit(train_X,train_Y)
    test_X, test_y = load_data_for_linear_SVMs('D:/E/CUONG/machine learning/20news-bydate/20news-test-tfidf.txt')
    
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y= predicted_y, expected_y=test_y)
    print('Accuracy:',accuracy)
def classifying_with_kernel_SVMs():
    train_X, train_Y = load_data_for_linear_SVMs('D:/E/CUONG/machine learning/20news-bydate/20news-train-tfidf.txt')
    
    from sklearn.svm import SVC
    classifier = SVC(C = 50.0, kernel = 'rbf', gamma = 0.1, tol = 0.001, verbose = True)
    classifier.fit(train_X,train_Y)
    test_X, test_y = load_data_for_linear_SVMs('D:/E/CUONG/machine learning/20news-bydate/20news-test-tfidf.txt')
    
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y= predicted_y, expected_y=test_y)
    print('Accuracy:',accuracy)
    
classifying_with_linear_SVMs()
classifying_with_kernel_SVMs()#Run very long
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:40:34 2022

@author: Admin
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import random
class Member:
    def __init__(self, r_d, label = None, doc_id = None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id
class Cluster:
    def __init__(self):
        self._centroid = None 
        self._members = []
    def reset_members(self):
        self._members = []
    def set_centroid(self, new_centroid):
        self._centroid = new_centroid
    def add_members(self, new_member):
        self._members.append(new_member)
class KMeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for i in range(self._num_clusters)]
        self._E = []       # List of centroids
        self._S = 0        # Overall similarity
    
    def load_data(self, data_path = 'D:/E/CUONG/machine learning/20news-bydate/20news-train-tfidf.txt'):
        # Get data from each line with label, doc_id, index & tfidf of its vocab
        def sparse_to_dense(sparse_r_d, vocab_size):
            # Init list size vocal size to store vocal 
            r_d = [0.0 for _ in range(vocab_size)]

            indices_and_tfidfs = sparse_r_d.split()
            for index_and_tfidf in indices_and_tfidfs:
                index = int(index_and_tfidf.split(':')[0])
                tfidf = float(index_and_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)    
                
        with open('D:/E/CUONG/machine learning/20news-bydate/20news-train-tfidf.txt') as f:
            data_lines = f.read().splitlines()
        with open('D:/E/CUONG/machine learning/20news-bydate/words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(data_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def random_init(self, seed_value):
        random.seed(seed_value)
        members = [member._r_d for member in self._data]

        pos = np.random.choice(len(self._data), self._num_clusters, replace=False)
        centroid = []
        for i in pos:
            centroid.append(members[i])
        self._E = centroid
        for i in range(self._num_clusters):
            self._clusters[i].set_centroid(centroid[i])

    def compute_similarity(self, member, centroid):

        return cosine_similarity([member._r_d], [centroid])

    def select_cluster_for(self, member):
        best_fit_cluster = None

        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_members(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        avg_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(avg_r_d ** 2))
        new_centroid = avg_r_d / sqrt_sum_sqr

        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False    

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)
        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
                self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1. 
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]
        for label in range(20):
            wk_cj = member_labels.count(label) * 1.
            cj = self._label_count[label]
            I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)
kmeans = KMeans(num_clusters=5)
kmeans.load_data()
kmeans.run(seed_value=36, criterion='similarity', threshold=1e-3)
print(kmeans.compute_purity(), kmeans.compute_NMI())
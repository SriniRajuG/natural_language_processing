from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import os


def read_text_data(text_data_dir):
    texts = []  # list of strings. Each element of this list is a document.
    labels_int_map = {}
    # A dictionary mapping label names (class names) to integer ids. Starting with 0
    labels = []  # list of integers
    # *texts* and *labels* will be of the same size.
    for dir_name in sorted(os.listdir(text_data_dir)):
        # loop over each directory (type of news, 0 to 19)
        # *dir_name* is label name / class name
        dir_path = os.path.join(text_data_dir, dir_name)
        if os.path.isdir(dir_path):
            label_id = len(labels_int_map)  # label_id is the integer corresponding to a class
            labels_int_map[dir_name] = label_id
            for file_name in sorted(os.listdir(dir_path)):
                if file_name.isdigit():
                    file_path = os.path.join(dir_path, file_name)
                    if sys.version_info < (3,):
                        fo = open(file_path)
                    else:
                        fo = open(file_path, encoding='latin-1')
                    txt = fo.read()
                    i = txt.find('\n\n')  # skip header
                    if 0 < i:
                        txt = txt[i:]
                    texts.append(txt)
                    fo.close()
                    labels.append(label_id)
    print('no. of documents', len(texts))
    print('no. of labels', len(labels_int_map))
    return texts, labels, labels_int_map


def main():

    use_hashing = False
    use_idf = True
    n_features = 10000
    n_components = 5000

    text_data_dir = '/home/osboxes/PycharmProjects/learn_neural_net/NLP/text_data/20_newsgroup'
    texts, labels, labels_int_map = read_text_data(text_data_dir)
    n_labels = len(labels_int_map)

    if use_hashing:
        if use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(analyzer='word', stop_words='english', lowercase=True,
                                       norm=None, binary=False,
                                       n_features=n_features)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(analyzer='word', stop_words='english',
                                           norm='l2', lowercase=True,
                                           binary=False, n_features=n_features)
    else:
        vectorizer = TfidfVectorizer(lowercase=True, max_df=0.5, max_features=n_features,
                                     min_df=2, stop_words='english', use_idf=use_idf)

    X = vectorizer.fit_transform(texts)
    # output is a scipy sparce csr matrix

    if n_components:
        print("Performing dimensionality reduction using LSA")
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
        print()


if __name__ == '__main__':
    main()

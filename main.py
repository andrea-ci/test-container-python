# -*- coding: utf-8 -*-
import argparse
from time import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def kmeans_benchmark(n_samples):

    n_classes = 3
    X, y = make_blobs(n_samples = n_samples, centers = n_classes, n_features = 4,
        random_state = 0)

    kmeans = KMeans(n_clusters = n_classes, random_state = 0).fit(X)
    labels = kmeans.labels_

    return y

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'K-means function for benchmark purpose.')
    parser.add_argument('nsamples', type = int, action = 'store',
        help = 'Number of samples to use')

    args = parser.parse_args()

    t_a = time()
    y = kmeans_benchmark(args.nsamples)
    t_b = time()

    print(f'Elapsed time: {t_b - t_a}')

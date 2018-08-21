# encoding=utf8
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import gc
from scipy.spatial.distance import pdist, squareform, is_valid_y, is_valid_dm, num_obs_y, num_obs_dm
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from numba import jit


def calculate_dispersion(x, labels):
    """
    Calculate the dispersion between points in each cluster
    Params:
        X: ndarry of shape (n_samples, n_features)
        labels: tha clusters labels from clustering function
    Returns: the sum of pairwaise distance for each k cluster
    """
    return np.sum(np.sum((1. / (2. * len(labels[labels == c]))) * pdist(x[x[:, -1] == c][:, :-1]).sum())
                  for c in np.unique(labels))


@jit
def clustering(X, method, k):
    """
    Clustering data into k clusters
    Params:
        X: ndarry of shape (n_samples, n_features)
        labels: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    """
    dist = pdist(X)
    l = linkage(dist, method=method)
    return fcluster(l, k, 'maxclust').reshape(-1, 1)


@jit
def gap_statistic(data, k, origLogW, r, f, method='ward', brefs=10):

    # Holder for reference dispersion results
    BWs = np.zeros(brefs)
    BlogWs = np.zeros(brefs)

    # For n references, generate random sample and perform kmeans getting resulting dispersion and gap of each loop
    for i in xrange(brefs):
        # Create new random reference set
        Breference = np.zeros((r, f))
        for n in xrange(f):
            xmin, xmax = np.min(data[:, n]), np.max(data[:, n])
            Breference[:, n] = np.random.uniform(xmin, xmax, size=r)

        labels = clustering(Breference, method, k)
        x = np.concatenate((Breference, labels), axis=1)
        BWs[i] = calculate_dispersion(x, labels)
        BlogWs[i] = np.log(BWs[i])

    # Calculate gap statistic
    Sd = np.std(BlogWs)
    ElogW = np.mean(BlogWs)
    Sk = np.sqrt(1.+1./brefs)*Sd
    gaps = ElogW - origLogW

    return gaps, ElogW, Sd, Sk


@jit
def swc(labels, distances):
    if len(np.unique(labels)) > 1:
        silhouette = round(silhouette_score(squareform(distances), labels, metric='precomputed'), 3)
        unique, counts = np.unique(labels, return_counts=True)
        stds = np.std(counts)
        armonicas = hmean(counts)
        armonicas_1 = 1 / np.sum(1.0/counts, axis=0)
    else:
        silhouette = 0.0
        stds = 'no'
        armonicas = 'no'
        armonicas_1 = 'no'

    return silhouette, stds, armonicas, armonicas_1


@jit
def ch(X, labels):
    if len(np.unique(labels)) > 1:
        return calinski_harabaz_score(X, labels)
    else:
        return 0.0
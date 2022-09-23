#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from matching.games import StableMarriage
import itertools

def rcors(m1, m2):
    # Single-cluster sorted correlations between to matrices
    m = np.zeros((len(m1), len(m2)))
    for i in range(len(m1)):
        for j in range(len(m2)):
            # "Correlation" between two clusters based on the distribution of their edge weights
            # Sort the edge weights then compute pearson correlation
            m[i, j], _ = pearsonr(np.sort(m1[i]), np.sort(m2[j]))
    return m


def double_center(m):
    # Center a matrix such that the sum of each row == 0, and the sum of each column == 0
    dim = len(m)
    m1 = m - np.repeat(np.mean(m, axis=0).reshape(-1, dim), dim, axis=0)
    m1 -= np.repeat(np.mean(m, axis=1).reshape(dim, -1), dim, axis=1)
    m1 += np.mean(m)
    return m1


def dcov(m1, m2):
    # Compute distance covariance of two double-centered matrices 
    return np.sqrt(np.sum(m1 * m2) / (len(m1) * len(m1)))


def dvar(m):
    # Compute distance variance of two double-centered matrices
    return np.sqrt(np.sum(m ** 2) / (len(m) * len(m)))


def dcor(m1, m2):
    # Compute distance correlation of two matrices
    m1 = double_center(m1)
    m2 = double_center(m2)
    return dcov(m1, m2) / np.sqrt(dvar(m1) * dvar(m2))


def match(m):
    # Perform Gale-Shapley stable matching. m[i, j] is the "similarity" between clusters i and j
    dim = len(m)
    men = {}
    for i in range(dim):
        men[i] = np.flip(np.argsort(m[i]))
    women = {}
    for i in range(dim):
        women[i] = np.flip(np.argsort(m[:, i]))
    game = StableMarriage.create_from_dictionaries(men, women)
    match = game.solve()
    result = [0] * len(m)
    for key in match:
        result[key.name] = match[key].name
    return result


def sym_perm_matrix(m):
    # Permute a matrix m such that the orders of rows and columns are changed in the same way
    dim = len(m)
    b = np.random.permutation(dim)
    m_p = np.zeros((dim, dim))
    map_back = [0] * dim
    for i in range(dim):
        map_back[b[i]] = i
        for j in range(dim):
            m_p[b[i]][b[j]] = m[i][j]
    # map_back[i] is the original row of the ith row of the new matrix 
    return m_p, map_back


def generate_batch(mus, sds, weights, n_cells=1000, n_genes=100):
    # Randomly generate some single-cell data by projecting a 2D mixture of Gaussians into a high-dimensional space
    n_groups = len(mus)
    labels = np.random.multinomial(1, weights, n_cells) 
    xs = np.zeros((n_cells, n_groups))
    ys = np.zeros((n_cells, n_groups))
    for i in range(n_groups):
        xs[:, i] = np.random.normal(mus[i][0], sds[i][0], n_cells)
        ys[:, i] = np.random.normal(mus[i][1], sds[i][1], n_cells)
        
    x = np.sum(xs * labels, axis=1).reshape(-1, 1)
    y = np.sum(ys * labels, axis=1).reshape(-1, 1)
    sample_1 = np.append(x, y, axis=1)
    # Project to high dimensional space
    proj = np.random.normal(0, 1, (2, n_genes))
    # Add random gene-specific batch effect and random noise
    cells_1 = np.dot(sample_1, proj) + np.random.normal(0, 1, (n_cells, n_genes))
    cells_1 += np.repeat(np.random.normal(0, 1, (1, n_genes)), n_cells, axis=0)
    labels = np.dot(labels, np.arange(n_groups).reshape(-1, 1))
    return cells_1, labels


def compute_distances(cells, labels, metric='euclidean'):
    # Compute pairwise average distances between clusters
    n_groups = len(labels) + 1
    matrix = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        cells_1 = cells[np.asarray(labels == i).reshape(-1)]
        for j in range(i + 1, n_groups):
            cells_2 = cells[np.asarray(labels == j).reshape(-1)]
            dist = np.mean(cdist(cells_1, cells_2, metric=metric))
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def match_two_experiments(cells_1, labels_1, cells_2, labels_2, metric='euclidean'):
    m1 = compute_distances(cells_1, labels_1, metric)
    m2 = compute_distances(cells_2, labels_2, metric)
#     match_res = match(rcors(m2, m1))
#     match_res, _ = brute_match(m1, m2)
    match_res = brute_match_perm(m1, m2)
    return match_res


def convert_to_ranks(m):
    lower = np.tril(m).flatten()
    lower = lower[lower != 0]
    s = list(np.sort(lower))
    ranks = [s.index(i) + 1 for i in lower]
    m2 = np.zeros(m.shape)
    lower = list(lower)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i][j] != 0:
                m2[i][j] = ranks[lower.index(m[i][j])]
    return m2


def brute_match(m1, m2, use_rank=False):
    if use_rank:
        m1 = convert_to_ranks(m1)
        m2 = convert_to_ranks(m2)
    maxd = 0
    max_map = None
    for i in range(2000):
        m2_p, m2_map = sym_perm_matrix(m2)
        d = dcor(m1, m2_p)
        if d > maxd:
            maxd = d
            max_map = m2_map
    return max_map, maxd


def brute_match_perm(m1, m2):
    maxd = 0
    max_map = None
    perms = list(itertools.permutations(np.arange(len(m1))))
    for p in perms:
        m2_p = m2[list(p)][:, list(p)]
        d = dcor(m1, m2_p)
        if d > maxd:
            maxd = d
            max_map = p
    return max_map, maxd

# def main():
#     mus = [[0, 0], [3, 3], [3, 0], [1.5, 6], [6, -1.5], [3, 10]]
#     sds = [[1, 1], [0.1, 0.1], [1, 1], [0.5, 0.1], [0.1, 0.5], [0.5, 0.5]]
#     weights = [[0.2, 0.2, 0.2, 0.1, 0.1, 0.2], [0.2, 0.25, 0.15, 0.1, 0.15, 0.15]]
#     n_iter = 100
#     accs = np.zeros(n_iter)     
#     for i in range(n_iter):
#         cells_1, labels_1 = generate_batch(mus, sds, weights[0])
#         cells_2, labels_2 = generate_batch(mus, sds, weights[1])
#         result = match_two_experiments(cells_1, labels_1, cells_2, labels_2)
#         accs[i] = np.sum([result[i] == i for i in range(len(mus))], dtype=float) / len(mus)
#     return accs
#     print("Accuracy: %.5f" % np.mean(accs))


# if __name__ == "__main__":
#     main()
          

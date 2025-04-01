"""
Functions to work with Hodge-Laplace operators
"""

import numpy as np
from itertools import combinations


def find_cliques(matrix, size_of_clique):
    """Find cliques of size size_of_clique on the adjacency matrix"""
    if size_of_clique == 1:
        cliques = [[i] for i in range(matrix.shape[0])]
    else:
        cliques = list()
        for potential_clique in combinations([i for i in range(matrix.shape[0])], size_of_clique):
            if all(matrix[a, b] != 0 for a, b in combinations(potential_clique, 2)):
                cliques.append(potential_clique)

    return cliques


def compute_k_weight_matrix(cliques_k, functional_connectome, k):
    """Find weighted matrix"""
    weight_matrix = np.zeros((len(cliques_k), len(cliques_k)))

    if k == 0:
        for i in range(functional_connectome.shape[0]):
            weight_matrix[i, i] = np.sum(functional_connectome[i][:])
    else:
        for i in range(len(cliques_k)):
            weight_matrix[i, i] = np.sum(
                np.array([functional_connectome[a, b] for a, b in combinations(cliques_k[i], 2)]))

    return weight_matrix


def find_boundary_matrix(cliques_k, cliques_km1):
    """Find boundary matrix"""
    boundary_matrix = np.zeros((len(cliques_km1), len(cliques_k)))
    if len(cliques_k) == 0:
        return boundary_matrix

    for i in range(len(cliques_km1)):
        for j in range(len(cliques_k)):
            if all(x in cliques_k[j] for x in cliques_km1[i]):
                indexes = [k for k in range(len(cliques_k[j]))]
                while cliques_k[j][indexes[-1]] in cliques_km1[i]:
                    indexes.pop()

                boundary_matrix[i, j] = (-1 if indexes[-1] % 2 != 0 else 1)

    return boundary_matrix


def compute_k_hodge_laplacian(functional_connectome, k, is_weighted):
    """Find k-Hodge Laplacian"""
    if k == 0:
        return compute_graph_laplacian(functional_connectome, is_weighted)

    cliques_k = find_cliques(functional_connectome, k + 1)
    cliques_kp1 = find_cliques(functional_connectome, k + 2)
    cliques_km1 = find_cliques(functional_connectome, k)

    if len(cliques_km1) == 0 or len(cliques_k) == 0:
        return np.array([])

    boundary_matrix_k = find_boundary_matrix(cliques_k, cliques_km1)
    boundary_matrix_kp1 = find_boundary_matrix(cliques_kp1, cliques_k)

    if is_weighted:
        weight_matrix_k = compute_k_weight_matrix(cliques_k, functional_connectome, k)
        weight_matrix_kp1 = compute_k_weight_matrix(cliques_kp1, functional_connectome, k + 1)
        weight_matrix_km1 = compute_k_weight_matrix(cliques_km1, functional_connectome, k - 1)
        weight_matrix_k_inv = np.linalg.inv(weight_matrix_k)
        for i in range(weight_matrix_km1.shape[0]):
            if weight_matrix_km1[i, i] != 1 and k == 1:
                weight_matrix_km1[i, i] -= 1
        weight_matrix_km1_inv = np.linalg.inv(weight_matrix_km1)
        for i in range(weight_matrix_km1.shape[0]):
            if weight_matrix_km1[i, i] == 1 and k == 1:
                weight_matrix_km1[i, i] -= 1
                weight_matrix_km1_inv[i, i] -= 1

        hodge_laplacian_matrix = (boundary_matrix_k.T @ weight_matrix_km1_inv @ boundary_matrix_k @ weight_matrix_k
                                  + weight_matrix_k_inv @ boundary_matrix_kp1 @ weight_matrix_kp1 @ boundary_matrix_kp1.T)
    else:
        hodge_laplacian_matrix = (boundary_matrix_k.T @ boundary_matrix_k
                                  + boundary_matrix_kp1 @ boundary_matrix_kp1.T)

    return hodge_laplacian_matrix


def compute_graph_laplacian(functional_connectome, is_weighted):
    """Find graph Laplacian"""
    if not is_weighted:
        functional_connectome = np.where(functional_connectome > 0, 1, 0)

    D = np.zeros(functional_connectome.shape)
    for i in range(functional_connectome.shape[0]):
        D[i, i] = np.sum(functional_connectome[i, :])

    return np.array(D - functional_connectome)


def find_eigenvalues_of_matrix(matrix):
    """Find the eigenvalues for the operator matrix"""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues.sort()
    eigenvalues = np.where(abs(eigenvalues - 0) < 1e-13, 0, eigenvalues)
    eigenvalues = np.real(eigenvalues)
    return eigenvalues


def find_betti_number(hodge_laplacian):
    """Find Betti numbers"""
    return hodge_laplacian.shape[1] - np.linalg.matrix_rank(hodge_laplacian)

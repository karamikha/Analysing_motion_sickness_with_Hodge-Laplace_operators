import numpy as np
from itertools import combinations


def FindCliques(matrix, size_of_clique):
    if size_of_clique == 1:
        cliques = [[i] for i in range(matrix.shape[0])]
    else:
        cliques = list()
        for potential_clique in combinations([i for i in range(matrix.shape[0])], size_of_clique):
            if all(matrix[a, b] != 0 for a, b in combinations(potential_clique, 2)):
                cliques.append(potential_clique)

    return cliques


def ComputeKWeightMatrix(cliques_k, functional_connectome, k):
    weight_matrix = np.zeros((len(cliques_k), len(cliques_k)))

    if k == 0:
        for i in range(functional_connectome.shape[0]):
            weight_matrix[i, i] = np.sum(functional_connectome[i][:])
    else:
        for i in range(len(cliques_k)):
            weight_matrix[i, i] = np.sum(np.array([functional_connectome[a, b] for a, b in combinations(cliques_k[i], 2)]))

    return weight_matrix


def FindBoundaryMatrix(cliques_k, cliques_km1):
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


def ComputeKHodgeLaplacian(functional_connectome, k, is_weighted):
    if k == 0:
        return ComputeGraphLaplacian(functional_connectome)

    cliques_k = FindCliques(functional_connectome, k + 1)
    cliques_kp1 = FindCliques(functional_connectome, k + 2)
    cliques_km1 = FindCliques(functional_connectome, k)

    if len(cliques_km1) == 0:
        return np.array([])

    boundary_matrix_k = FindBoundaryMatrix(cliques_k, cliques_km1)
    boundary_matrix_kp1 = FindBoundaryMatrix(cliques_kp1, cliques_k)

    if is_weighted:
        weight_matrix_k = ComputeKWeightMatrix(cliques_k, functional_connectome, k)
        weight_matrix_kp1 = ComputeKWeightMatrix(cliques_kp1, functional_connectome, k + 1)
        weight_matrix_km1 = ComputeKWeightMatrix(cliques_km1, functional_connectome, k - 1)
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


def ComputeGraphLaplacian(functional_connectome):
    D = np.zeros(functional_connectome.shape)
    for i in range(functional_connectome.shape[0]):
        D[i, i] = np.sum(functional_connectome[i, :])

    return np.array(D - functional_connectome)


def FindEigenValuesOfMatrix(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues.sort()
    eigenvalues = np.where(abs(eigenvalues - 0) < 1e-13, 0, eigenvalues)
    eigenvalues = np.real(eigenvalues)
    return eigenvalues


def FindBettiNumber(hodge_laplacian):
    return hodge_laplacian.shape[1] - np.linalg.matrix_rank(hodge_laplacian)

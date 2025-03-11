import numpy as np


def ComputeGraphLaplacian(functional_connectome):
    D = np.zeros(functional_connectome.shape)
    for i in range(functional_connectome.shape[0]):
        D[i, i] = np.sum(functional_connectome[i][:])

    return D - functional_connectome


def FindEigenValuesOfMatrix(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues.sort()
    eigenvalues = np.where(abs(eigenvalues - 0) < 1e-13, 0, eigenvalues)
    eigenvalues = np.real(eigenvalues)
    return eigenvalues

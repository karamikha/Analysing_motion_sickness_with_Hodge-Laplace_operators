"""
Check graph Laplacian for the cumulative data about the active treatment with BCV
"""

from hodge_laplacians_functions import *
from visualize.visualization import visualize_graph, visualize_func_connectome

EEG_data_before_treatment = np.loadtxt("EEG_data/EEG_data_before_treatment.txt")
EEG_data_after_treatment = np.loadtxt("EEG_data/EEG_data_after_treatment.txt")

channels_names = ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"]

corr_matrix_before_treatment = np.abs(np.corrcoef(EEG_data_before_treatment))
corr_matrix_before_treatment = np.where(corr_matrix_before_treatment > 0.7, corr_matrix_before_treatment, 0)

corr_matrix_after_treatment = np.abs(np.corrcoef(EEG_data_after_treatment))
corr_matrix_after_treatment = np.where(corr_matrix_after_treatment > 0.7, corr_matrix_after_treatment, 0)

graph_laplacian_before_treatment = compute_graph_laplacian(corr_matrix_before_treatment)

print(
    f"eigenvalues before treatment for the graph Laplacian: {find_eigenvalues_of_matrix(graph_laplacian_before_treatment)}, Betti number for the graph Laplacian: {find_betti_number(graph_laplacian_before_treatment)}")

graph_laplacian_after_treatment = compute_graph_laplacian(corr_matrix_after_treatment)

print(
    f"eigenvalues after treatment for the graph Laplacian: {find_eigenvalues_of_matrix(graph_laplacian_after_treatment)}, Betti number for the graph Laplacian: {find_betti_number(graph_laplacian_after_treatment)}")

# drawing graph
visualize_graph(corr_matrix_before_treatment, channels_names, "Визуализация графа до лечения")
visualize_graph(corr_matrix_after_treatment, channels_names, "Визуализация графа после лечения")

# drawing functional connectome
visualize_func_connectome(corr_matrix_before_treatment, channels_names, "Матрица корреляций до лечения")
visualize_func_connectome(corr_matrix_after_treatment, channels_names, "Матрица корреляций после лечения")

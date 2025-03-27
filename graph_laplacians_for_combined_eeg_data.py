from hodge_laplacians_functions import *
from visualize.visualization import VisualizeGraph, VisualizeFuncConnectome

EEG_data_before_treatment = np.loadtxt("EEG_data/EEG_data_before_treatment.txt")
EEG_data_after_treatment = np.loadtxt("EEG_data/EEG_data_after_treatment.txt")

channels_names = ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"]

corr_matrix_before_treatment = np.abs(np.corrcoef(EEG_data_before_treatment))
corr_matrix_before_treatment = np.where(corr_matrix_before_treatment > 0.7, corr_matrix_before_treatment, 0)

corr_matrix_after_treatment = np.abs(np.corrcoef(EEG_data_after_treatment))
corr_matrix_after_treatment = np.where(corr_matrix_after_treatment > 0.7, corr_matrix_after_treatment, 0)

graph_laplacian_before_treatment = ComputeGraphLaplacian(corr_matrix_before_treatment)

print(f"eigenvalues before treatment for the graph Laplacian: {FindEigenValuesOfMatrix(graph_laplacian_before_treatment)}, Betti number for the graph Laplacian: {FindBettiNumber(graph_laplacian_before_treatment)}")

graph_laplacian_after_treatment = ComputeGraphLaplacian(corr_matrix_after_treatment)

print(f"eigenvalues after treatment for the graph Laplacian: {FindEigenValuesOfMatrix(graph_laplacian_after_treatment)}, Betti number for the graph Laplacian: {FindBettiNumber(graph_laplacian_after_treatment)}")

# drawing graph
VisualizeGraph(corr_matrix_before_treatment, channels_names, "Визуализация графа до лечения")
VisualizeGraph(corr_matrix_after_treatment, channels_names, "Визуализация графа после лечения")

# drawing functional connectome
VisualizeFuncConnectome(corr_matrix_before_treatment, channels_names, "Матрица корреляций до лечения")
VisualizeFuncConnectome(corr_matrix_after_treatment, channels_names, "Матрица корреляций после лечения")

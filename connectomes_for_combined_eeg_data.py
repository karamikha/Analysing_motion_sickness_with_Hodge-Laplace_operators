import numpy as np

EEG_data_before_treatment = np.loadtxt("EEG_data/EEG_data_before_treatment.txt")
EEG_data_after_treatment = np.loadtxt("EEG_data/EEG_data_after_treatment.txt")

channels_names = ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"]

corr_matrix_before_treatment = np.corrcoef(EEG_data_before_treatment)
corr_matrix_before_treatment = np.where(corr_matrix_before_treatment > 0.7, corr_matrix_before_treatment, 0)

corr_matrix_after_treatment = np.corrcoef(EEG_data_after_treatment)
corr_matrix_after_treatment = np.where(corr_matrix_after_treatment > 0.7, corr_matrix_after_treatment, 0)

from HodgeLaplaciansFunctions import *

graph_laplacian_before_treatment = ComputeGraphLaplacian(corr_matrix_before_treatment)

print(f"eigenvalues before treatment: {FindEigenValuesOfMatrix(graph_laplacian_before_treatment)}")

graph_laplacian_after_treatment = ComputeGraphLaplacian(corr_matrix_after_treatment)

print(f"eigenvalues after treatment: {FindEigenValuesOfMatrix(graph_laplacian_after_treatment)}")

# drawing graph
from visualize.visualization import *

VisualizeGraph(corr_matrix_before_treatment, channels_names)
VisualizeGraph(corr_matrix_after_treatment, channels_names)

# drawing functional connectome
VisualizeFuncConnectome(corr_matrix_before_treatment, channels_names)
VisualizeFuncConnectome(corr_matrix_after_treatment, channels_names)

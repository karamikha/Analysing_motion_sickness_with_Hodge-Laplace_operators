from pathlib import Path
import numpy as np
from HodgeLaplaciansFunctions import ComputeGraphLaplacian, FindEigenValuesOfMatrix

q_good_treatment = 0

for i in range(21):
    path_for_easy_files = Path(f"EEG_data/BCV/Active/{i + 1}")

    eigenvalues_before_treatment = np.array([])
    eigenvalues_after_treatment = np.array([])
    for file in path_for_easy_files.rglob("*.easy"):
        if file.is_file():
            EEG_data = np.loadtxt(f"EEG_data/BCV/Active/{i + 1}/" + file.name)
            EEG_data = EEG_data[:, :8].transpose()
            corr_matrix = np.corrcoef(EEG_data)
            corr_matrix = np.where(corr_matrix > 0.7, corr_matrix, 0)
            graph_laplacian = ComputeGraphLaplacian(corr_matrix)
            if file.name[-6] == "1":
                eigenvalues_before_treatment = FindEigenValuesOfMatrix(graph_laplacian)
            elif file.name[-6] == "2":
                eigenvalues_after_treatment = FindEigenValuesOfMatrix(graph_laplacian)

    if eigenvalues_before_treatment.size == 0 or eigenvalues_after_treatment.size == 0:
        continue

    print(
        f"Participant {i + 1} before treatment, eigenvalues: {eigenvalues_before_treatment}, zero eigenvalues = {np.sum(eigenvalues_before_treatment == 0)}, avg(eigenvalues): {np.mean(eigenvalues_before_treatment)}")
    print(
        f"Participant {i + 1} after treatment, eigenvalues: {eigenvalues_after_treatment}, zero eigenvalues = {np.sum(eigenvalues_after_treatment == 0)}, avg(eigenvalues): {np.mean(eigenvalues_after_treatment)}")
    print(
        f"delta zero eigenvalues: {np.sum(eigenvalues_before_treatment == 0) - np.sum(eigenvalues_after_treatment == 0)}")
    print("\n")

    if np.mean(eigenvalues_before_treatment) < np.mean(eigenvalues_after_treatment) and np.sum(eigenvalues_before_treatment == 0) - np.sum(eigenvalues_after_treatment == 0) > 0:
        q_good_treatment += 1

print(f"number of participants who have been helped by the treatment: {q_good_treatment}")

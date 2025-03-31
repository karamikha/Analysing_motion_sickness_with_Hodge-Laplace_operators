import numpy as np
from scipy import stats as sts
from pathlib import Path
from hodge_laplacians_functions import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import confusion_matrix


def FindVectorsOfEigenvalues(method, way, k, threshold):
    vectors_of_eigenvalues_after_treatment_list = []
    for i in range(21):
        path_for_easy_files = Path(f"../EEG_data/{method}/{way}/{i + 1}")

        files = list(path_for_easy_files.rglob("*.easy"))
        if len(files) == 2:

            files.sort(key=lambda f: int(f.name[-6]))

            for file in files:
                if file.is_file():
                    EEG_data = np.loadtxt(f"../EEG_data/{method}/{way}/{i + 1}/" + file.name)
                    EEG_data = EEG_data[:, :8].transpose()
                    corr_matrix = np.abs(np.corrcoef(EEG_data))
                    corr_matrix = np.where(corr_matrix > threshold, corr_matrix, 0)

                    graph_laplacian = ComputeKHodgeLaplacian(corr_matrix, k, True)

                    eigenvalues = FindEigenValuesOfMatrix(graph_laplacian)

                    if file.name[-6] == "2":
                        vectors_of_eigenvalues_after_treatment_list.append(eigenvalues)

    return vectors_of_eigenvalues_after_treatment_list


def InvestigateHypoForClasters(real_groups, predicted_groups):
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    p_value = sts.chi2_contingency(conf_matrix)[1]
    alpha = 0.05
    if p_value < alpha:
        print(f"Clustering and real groups are interrelated, p-value = {p_value}")
    else:
        print(f"Clustering and real groups are independent, p-value = {p_value}")


def FindErrorsRates(real_groups, predicted_groups):
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    TN, FP, FN, TP = conf_matrix.ravel()
    error_1_rate = FP / (FP + TN)
    error_2_rate = FN / (FN + TP)

    return error_1_rate, error_2_rate


for threshold in [0.7, 0.5, 0.3]:
    print(f"Threshold {threshold}:")
    for el in ["BCV", "tACS"]:
        print(f"{el}:")

        for k in (0, 1):
            print(f"{k}-Hodge Laplacian:")
            vectors_of_eigenvalues_for_active = FindVectorsOfEigenvalues(el, "Active", k, threshold)
            vectors_of_eigenvalues_for_sham = FindVectorsOfEigenvalues(el, "Sham", k, threshold)

            vectors_of_eigenvalues_medians_for_active = [np.median(el) for el in vectors_of_eigenvalues_for_active]
            vectors_of_eigenvalues_medians_for_sham = [np.median(el) for el in vectors_of_eigenvalues_for_sham]
            vectors_of_eigenvalues_sums_for_active = [np.sum(el) for el in vectors_of_eigenvalues_for_active]
            vectors_of_eigenvalues_sums_for_sham = [np.sum(el) for el in vectors_of_eigenvalues_for_sham]
            vectors_of_eigenvalues_means_for_active = [np.mean(el) for el in vectors_of_eigenvalues_for_active]
            vectors_of_eigenvalues_means_for_sham = [np.mean(el) for el in vectors_of_eigenvalues_for_sham]
            vectors_of_eigenvalues_vars_for_active = [np.var(el) for el in vectors_of_eigenvalues_for_active]
            vectors_of_eigenvalues_vars_for_sham = [np.var(el) for el in vectors_of_eigenvalues_for_sham]

            _, p_value = sts.mannwhitneyu(vectors_of_eigenvalues_medians_for_active,
                                          vectors_of_eigenvalues_medians_for_sham, alternative="two-sided")
            alpha = 0.05
            if p_value < 0.05:
                print(f"Median. Spectrа of eigenvalues for active and sham treatment have differences, p-value = {p_value}")
            else:
                print(f"Median. Spectrа of eigenvalues for active and sham treatment are interrelated, p-value = {p_value}")

            _, p_value = sts.mannwhitneyu(vectors_of_eigenvalues_sums_for_active,
                                          vectors_of_eigenvalues_sums_for_sham, alternative="two-sided")
            alpha = 0.05
            if p_value < 0.05:
                print(f"Sum. Spectrа of eigenvalues for active and sham treatment have differences, p-value = {p_value}")
            else:
                print(f"Sum. Spectrа of eigenvalues for active and sham treatment are interrelated, p-value = {p_value}")

            _, p_value = sts.mannwhitneyu(vectors_of_eigenvalues_means_for_active,
                                          vectors_of_eigenvalues_means_for_sham, alternative="two-sided")
            alpha = 0.05
            if p_value < 0.05:
                print(f"Mean. Spectrа of eigenvalues for active and sham treatment have differences, p-value = {p_value}")
            else:
                print(f"Mean. Spectrа of eigenvalues for active and sham treatment are interrelated, p-value = {p_value}")

            _, p_value = sts.mannwhitneyu(vectors_of_eigenvalues_vars_for_active,
                                          vectors_of_eigenvalues_vars_for_sham, alternative="two-sided")
            alpha = 0.05
            if p_value < 0.05:
                print(f"Var. Spectrа of eigenvalues for active and sham treatment have differences, p-value = {p_value}")
            else:
                print(f"Var. Spectrа of eigenvalues for active and sham treatment are interrelated, p-value = {p_value}")

            if k == 0:
                vectors_of_eigenvalues_for_bcv = np.array(vectors_of_eigenvalues_for_active)
                vectors_of_eigenvalues_for_tacs = np.array(vectors_of_eigenvalues_for_sham)
                vectors_of_eigenvalues = np.vstack(
                    [vectors_of_eigenvalues_for_bcv, vectors_of_eigenvalues_for_tacs])
                real_groups = np.array(
                    [0] * len(vectors_of_eigenvalues_for_bcv) + [1] * len(
                        vectors_of_eigenvalues_for_tacs))
                spec_clust = SpectralClustering(n_clusters=2, random_state=13)
                predicted_groups = spec_clust.fit_predict(vectors_of_eigenvalues)

                print(f"Accuracy: {np.mean(predicted_groups == real_groups)}")

                error_1_rate, error_2_rate = FindErrorsRates(real_groups, predicted_groups)
                print(f"Type 1 error rate: {error_1_rate}, type 2 error rate: {error_2_rate}")

                InvestigateHypoForClasters(real_groups, predicted_groups)

            print()

        print()
    print("\n\n")
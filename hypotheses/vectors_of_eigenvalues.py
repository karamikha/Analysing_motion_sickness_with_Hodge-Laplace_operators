"""
Hypotheses for vectors of eigenvalues before and after treatment
"""

from scipy import stats as sts
from pathlib import Path
from hodge_laplacians_functions import *
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix


def find_vectors_of_eigenvalues(method, way, threshold):
    """Find vectors of eigenvalues"""
    vectors_of_eigenvalues_before_treatment_list = []
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

                    graph_laplacian = compute_k_hodge_laplacian(corr_matrix, 0, True)

                    eigenvalues = find_eigenvalues_of_matrix(graph_laplacian)

                    if file.name[-6] == "1":
                        vectors_of_eigenvalues_before_treatment_list.append(eigenvalues)
                    else:
                        vectors_of_eigenvalues_after_treatment_list.append(eigenvalues)

    return vectors_of_eigenvalues_before_treatment_list, vectors_of_eigenvalues_after_treatment_list


def find_real_and_predicted_groups(vectors_of_eigenvalues_before_treatment_list,
                                   vectors_of_eigenvalues_after_treatment_list):
    """Find clusters and real groups by spectral clustering"""
    vectors_of_eigenvalues_before_treatment_list = np.array(vectors_of_eigenvalues_before_treatment_list)
    vectors_of_eigenvalues_after_treatment_list = np.array(vectors_of_eigenvalues_after_treatment_list)
    vectors_of_eigenvalues_treatment_list = np.vstack(
        [vectors_of_eigenvalues_before_treatment_list, vectors_of_eigenvalues_after_treatment_list])

    real_groups = np.array(
        [0] * len(vectors_of_eigenvalues_before_treatment_list) + [1] * len(
            vectors_of_eigenvalues_after_treatment_list))
    spec_clust = SpectralClustering(n_clusters=2, random_state=13)
    predicted_groups = spec_clust.fit_predict(vectors_of_eigenvalues_treatment_list)

    return real_groups, predicted_groups


def find_errors_rates(real_groups, predicted_groups):
    """Find rates of type I and type II errors"""
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    tn, fp, fn, tp = conf_matrix.ravel()
    error_1_rate = fp / (fp + tn)
    error_2_rate = fn / (fn + tp)

    return error_1_rate, error_2_rate


def investigate_hypo_for_clusters(real_groups, predicted_groups):
    """Investigate the hypothesis about clusters and real groups using chi2-test"""
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    p_value = sts.chi2_contingency(conf_matrix)[1]
    alpha = 0.05
    if p_value < alpha:
        print(f"Clustering and real groups are interrelated, p-value = {p_value}")
    else:
        print(f"Clustering and real groups are independent, p-value = {p_value}")


for threshold in [0.7, 0.5, 0.3]:
    print(f"Threshold {threshold}:")
    for el in ["BCV", "tACS"]:
        print(f"{el}:")

        vectors_of_eigenvalues_before_active_treatment_list, vectors_of_eigenvalues_after_active_treatment_list = find_vectors_of_eigenvalues(
            el, "Active", threshold)
        vectors_of_eigenvalues_before_sham_treatment_list, vectors_of_eigenvalues_after_sham_treatment_list = find_vectors_of_eigenvalues(
            el, "Sham", threshold)

        print("Active:")
        real_groups, predicted_groups = find_real_and_predicted_groups(
            vectors_of_eigenvalues_before_active_treatment_list,
            vectors_of_eigenvalues_after_active_treatment_list)

        accuracy = np.mean(predicted_groups == real_groups)
        print(f"Clusterization accuracy: {accuracy}")

        error_1_rate, error_2_rate = find_errors_rates(real_groups, predicted_groups)
        print(f"Type 1 error rate: {error_1_rate}, type 2 error rate: {error_2_rate}")

        investigate_hypo_for_clusters(real_groups, predicted_groups)

        print()
        print("Sham:")
        real_groups, predicted_groups = find_real_and_predicted_groups(
            vectors_of_eigenvalues_before_sham_treatment_list,
            vectors_of_eigenvalues_after_sham_treatment_list)

        accuracy = np.mean(predicted_groups == real_groups)
        print(f"Clusterization accuracy: {accuracy}")

        error_1_rate, error_2_rate = find_errors_rates(real_groups, predicted_groups)
        print(f"Type 1 error rate: {error_1_rate}, type 2 error rate: {error_2_rate}")

        investigate_hypo_for_clusters(real_groups, predicted_groups)

        print("\n")

    print("\n\n")

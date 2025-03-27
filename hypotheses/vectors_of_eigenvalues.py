"""
H0: the treatment hasn't reduced the number of components
"""
from scipy import stats as sts
from pathlib import Path
from hodge_laplacians_functions import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def FindVectorsOfEigenvalues(method, way):
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
                    corr_matrix = np.where(corr_matrix > 0.7, corr_matrix, 0)

                    graph_laplacian = ComputeKHodgeLaplacian(corr_matrix, 0, True)

                    eigenvalues = FindEigenValuesOfMatrix(graph_laplacian)

                    if file.name[-6] == "1":
                        vectors_of_eigenvalues_before_treatment_list.append(eigenvalues)
                    else:
                        vectors_of_eigenvalues_after_treatment_list.append(eigenvalues)

    return vectors_of_eigenvalues_before_treatment_list, vectors_of_eigenvalues_after_treatment_list

def FindRealAndPredictedGroups(vectors_of_eigenvalues_before_treatment_list, vectors_of_eigenvalues_after_treatment_list):
    vectors_of_eigenvalues_before_treatment_list = np.array(vectors_of_eigenvalues_before_treatment_list)
    vectors_of_eigenvalues_after_treatment_list = np.array(vectors_of_eigenvalues_after_treatment_list)
    vectors_of_eigenvalues_treatment_list = np.vstack(
        [vectors_of_eigenvalues_before_treatment_list, vectors_of_eigenvalues_after_treatment_list])

    real_groups = np.array(
        [0] * len(vectors_of_eigenvalues_before_treatment_list) + [1] * len(
            vectors_of_eigenvalues_after_treatment_list))
    kmeans = KMeans(n_clusters=2, random_state=13)
    predicted_groups = kmeans.fit_predict(vectors_of_eigenvalues_treatment_list)

    return real_groups, predicted_groups

def FindErrorsRates(real_groups, predicted_groups):
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    TN, FP, FN, TP = conf_matrix.ravel()
    error_1_rate = FP / (FP + TN)
    error_2_rate = FN / (FN + TP)

    return error_1_rate, error_2_rate

def InvestigateHypoForClasters(real_groups, predicted_groups):
    conf_matrix = confusion_matrix(real_groups, predicted_groups)
    p_value = sts.chi2_contingency(conf_matrix)[1]
    alpha = 0.05
    if p_value < alpha:
        print(f"Clustering and real groups are interrelated, p-value = {p_value}")
    else:
        print(f"Clustering and real groups are independent, p-value = {p_value}")


for el in ["BCV", "tACS"]:
    print(f"{el}:")

    vectors_of_eigenvalues_before_active_treatment_list, vectors_of_eigenvalues_after_active_treatment_list = FindVectorsOfEigenvalues(el, "Active")
    vectors_of_eigenvalues_before_sham_treatment_list, vectors_of_eigenvalues_after_sham_treatment_list = FindVectorsOfEigenvalues(el, "Sham")


    print("Active:")
    real_groups, predicted_groups = FindRealAndPredictedGroups(vectors_of_eigenvalues_before_active_treatment_list, vectors_of_eigenvalues_after_active_treatment_list)

    accuracy = np.mean(predicted_groups == real_groups)
    print(f"Clusterization accuracy: {accuracy}")

    error_1_rate, error_2_rate = FindErrorsRates(real_groups, predicted_groups)
    print(f"Type 1 error rate: {error_1_rate}, type 2 error rate: {error_2_rate}")

    InvestigateHypoForClasters(real_groups, predicted_groups)


    print()
    print("Sham:")
    real_groups, predicted_groups = FindRealAndPredictedGroups(vectors_of_eigenvalues_before_sham_treatment_list,
                                                               vectors_of_eigenvalues_after_sham_treatment_list)

    accuracy = np.mean(predicted_groups == real_groups)
    print(f"Clusterization accuracy: {accuracy}")

    error_1_rate, error_2_rate = FindErrorsRates(real_groups, predicted_groups)
    print(f"Type 1 error rate: {error_1_rate}, type 2 error rate: {error_2_rate}")

    InvestigateHypoForClasters(real_groups, predicted_groups)

    print("\n")

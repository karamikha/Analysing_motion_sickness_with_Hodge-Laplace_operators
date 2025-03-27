from scipy import stats as sts
from pathlib import Path
from hodge_laplacians_functions import *
import matplotlib.pyplot as plt


def FindQConnectedComponents(method, way):
    q_connected_comps_before_treatment_list = []
    q_connected_comps_after_treatment_list = []
    numbers_of_participants_for_treatment = []
    for i in range(21):
        path_for_easy_files = Path(f"../EEG_data/{method}/{way}/{i + 1}")

        files = list(path_for_easy_files.rglob("*.easy"))
        if len(files) == 2:
            numbers_of_participants_for_treatment.append(i + 1)

            files.sort(key=lambda f: int(f.name[-6]))

            for file in files:
                if file.is_file():
                    EEG_data = np.loadtxt(f"../EEG_data/{method}/{way}/{i + 1}/" + file.name)
                    EEG_data = EEG_data[:, :8].transpose()
                    corr_matrix = np.abs(np.corrcoef(EEG_data))
                    corr_matrix = np.where(corr_matrix > 0.7, corr_matrix, 0)

                    graph_laplacian = ComputeKHodgeLaplacian(corr_matrix, 0, True)

                    betti_number = FindBettiNumber(graph_laplacian)

                    if file.name[-6] == "1":
                        q_connected_comps_before_treatment_list.append(int(betti_number))
                    else:
                        q_connected_comps_after_treatment_list.append(int(betti_number))

    return q_connected_comps_before_treatment_list, q_connected_comps_after_treatment_list, numbers_of_participants_for_treatment


def InvestigateHypoForQConnComps(q_connected_comps_before_treatment_list, q_connected_comps_after_treatment_list, way):
    p_value = sts.wilcoxon(q_connected_comps_before_treatment_list,
                           q_connected_comps_after_treatment_list, alternative="less").pvalue
    alpha = 0.05
    if p_value < alpha:
        print(
            f"The {way} treatment has reduced the number of components (type 1 error is possible), p-value = {p_value}")
    else:
        print(
            f"The {way} treatment hasn't reduced the number of components (type 2 error is possible), p-value = {p_value}")


for el in ["BCV", "tACS"]:
    print(f"{el}:")

    q_connected_comps_before_active_treatment_list, q_connected_comps_after_active_treatment_list, numbers_of_participants_for_active_treatment = FindQConnectedComponents(
        el, "Active")
    InvestigateHypoForQConnComps(q_connected_comps_before_active_treatment_list,
                                 q_connected_comps_after_active_treatment_list, "Active")

    q_connected_comps_before_sham_treatment_list, q_connected_comps_after_sham_treatment_list, numbers_of_participants_for_sham_treatment = FindQConnectedComponents(
        el, "Sham")
    InvestigateHypoForQConnComps(q_connected_comps_before_sham_treatment_list,
                                 q_connected_comps_after_sham_treatment_list, "Sham")
    print()

    fig, axs = plt.subplots(figsize=(8, 6), ncols=1, nrows=2)
    axs[0].plot(numbers_of_participants_for_active_treatment, q_connected_comps_before_active_treatment_list,
                marker="o", color="blue",
                label="Активное лечение")
    axs[0].plot(numbers_of_participants_for_active_treatment, q_connected_comps_after_active_treatment_list, marker="o",
                color="red",
                label="Фиктивное лечение")
    axs[0].set_title(f"Число компонент до/после активного лечения для {el}")
    axs[0].set_xlabel("Номер испытуемого")
    axs[0].set_ylabel("Число компонент связности")
    axs[0].legend()

    axs[1].plot(numbers_of_participants_for_sham_treatment, q_connected_comps_before_sham_treatment_list, marker="o",
                color="blue",
                label="Активное лечение")
    axs[1].plot(numbers_of_participants_for_sham_treatment, q_connected_comps_after_sham_treatment_list, marker="o",
                color="red",
                label="Фиктивное лечение")
    axs[1].set_title(f"Число компонент до/после фиктивного лечения для {el}")
    axs[1].set_xlabel("Номер испытуемого")
    axs[1].set_ylabel("Число компонент связности")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

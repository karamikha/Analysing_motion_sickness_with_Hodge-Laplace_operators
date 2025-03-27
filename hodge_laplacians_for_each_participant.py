from pathlib import Path
from hodge_laplacians_functions import *
from visualize.visualization import VisualizeGraph

q_less_connected_comps = 0
q_chordal_before_treatment = 0
q_chordal_after_treatment = 0
q_participants = 0
for i in range(21):
    path_for_easy_files = Path(f"EEG_data/BCV/Active/{i + 1}")

    files = list(path_for_easy_files.rglob("*.easy"))
    if len(files) == 2:
        q_participants += 1

        files.sort(key=lambda f: int(f.name[-6]))
        print(f"Participant {i + 1}:\n")

        q_less_connected_comps_for_part = 0
        for file in files:
            if file.is_file():
                EEG_data = np.loadtxt(f"EEG_data/BCV/Active/{i + 1}/" + file.name)
                EEG_data = EEG_data[:, :8].transpose()
                corr_matrix = np.abs(np.corrcoef(EEG_data))
                corr_matrix = np.where(corr_matrix > 0.7, corr_matrix, 0)
                VisualizeGraph(corr_matrix, ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"], "")

                k = 0
                while True:
                    k_hodge_laplacian = ComputeKHodgeLaplacian(corr_matrix, k, True)
                    if k_hodge_laplacian.size == 0:
                        if k == 1:
                            if file.name[-6] == "1":
                                q_chordal_before_treatment += 1
                            else:
                                q_chordal_after_treatment += 1
                        break

                    eigenvalues = FindEigenValuesOfMatrix(k_hodge_laplacian)
                    betti_number = FindBettiNumber(k_hodge_laplacian)

                    if file.name[-6] == "1":
                        if k == 0:
                            q_less_connected_comps_for_part += betti_number
                        if k == 1 and betti_number == 0:
                            q_chordal_before_treatment += 1

                        print(
                            f"Before treatment, {k}-Hodge Laplacian, Betti number{" (number of connected components)" if k == 0 else ""} = {betti_number}, avg(eigenvalues): {np.mean(eigenvalues)}")
                    else:
                        if k == 0:
                            q_less_connected_comps_for_part -= betti_number
                        if k == 1 and betti_number == 0:
                            q_chordal_after_treatment += 1

                        print(
                            f"After treatment, {k}-Hodge Laplacian, Betti number{" (number of connected components)" if k == 0 else ""} = {betti_number}, avg(eigenvalues): {np.mean(eigenvalues)}")

                    k += 1
                print()

        print("\n")

        if q_less_connected_comps_for_part > 0:
            q_less_connected_comps += 1

print("Some useful info:")
print(f"- number of participants who have been helped by the treatment: {q_less_connected_comps}")
print(f"- {q_chordal_before_treatment} of {q_participants} graphs for participants before treatment are chordal, {q_chordal_after_treatment} of {q_participants} graphs for participants after treatment are chordal")

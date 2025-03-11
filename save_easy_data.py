import numpy as np
from pathlib import Path

EEG_data_before_treatment = []
EEG_data_after_treatment = []
for i in range(21):
    path_for_easy_files = Path(f"EEG_data/BCV/Active/{i + 1}")
    for file in path_for_easy_files.rglob("*.easy"):
        if file.is_file():
            data = np.loadtxt(f"EEG_data/BCV/Active/{i + 1}/"+file.name)
            data = data[:, :8].transpose()
            if file.name[-6] == "1":
                if len(EEG_data_before_treatment) == 0:
                    EEG_data_before_treatment = data
                else:
                    EEG_data_before_treatment = np.hstack((EEG_data_before_treatment, data))
            else:
                if len(EEG_data_after_treatment) == 0:
                    EEG_data_after_treatment = data
                else:
                    EEG_data_after_treatment = np.hstack((EEG_data_after_treatment, data))

np.savetxt('EEG_data/EEG_data_before_treatment.txt', EEG_data_before_treatment)
np.savetxt('EEG_data/EEG_data_after_treatment.txt', EEG_data_after_treatment)

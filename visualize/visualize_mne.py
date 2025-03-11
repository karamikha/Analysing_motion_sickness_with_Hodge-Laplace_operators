import numpy as np

data = np.loadtxt("../EEG_data/BCV/Active/1/20220620113331_1-1.easy")

EEG_data = data[:, :8].transpose()

channels_names = ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"]

from visualize.visualization import VisualizeMNE

VisualizeMNE(EEG_data, channels_names)

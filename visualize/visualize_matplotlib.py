import numpy as np

data = np.loadtxt("../EEG_data/BCV/Active/1/20220620113331_1-1.easy")

EEG_data = data[:, :8]
times = data[:, -1]
times_in_sec = (times - times[0]) / 1000

channels_names = ["Fz", "Pz", "P3", "P4", "Cz", "EXT", "CP5", "CP6"]

from visualize.visualization import VisualizeMatplotlib

VisualizeMatplotlib(EEG_data, channels_names, times)

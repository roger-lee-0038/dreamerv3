import numpy as np
from pymoo.indicators.hv import Hypervolume
import pickle
import matplotlib
from matplotlib import pyplot as plt

F1 = np.loadtxt("resFsar.txt")
with open("histFsar.pkl", "rb") as F1_handler:
    hist_F1 = pickle.load(F1_handler)

F2 = np.loadtxt("resFsar_himoss.txt")
with open("histFsar_himoss.pkl", "rb") as F2_handler:
    hist_F2 = pickle.load(F2_handler)

F_all = np.vstack([F1, F2])

approx_ideal = F_all.min(axis=0)
approx_nadir = F_all.max(axis=0)

metric = Hypervolume(ref_point= np.array([1.1, 1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

hv1 = [metric.do(_F) for _F in hist_F1]
hv2 = [metric.do(_F) for _F in hist_F2]

plt.figure(figsize=(7, 5))
plt.plot(range(len(hist_F1)), hv1,  color='blue', lw=0.7, label="NSGAII")
plt.plot(range(len(hist_F2)), hv2,  color='red', lw=0.7, label="HiMOSS")
plt.title("Convergence")
plt.xlabel("Iterations")
plt.ylabel("Hypervolume")
plt.legend()
plt.show()

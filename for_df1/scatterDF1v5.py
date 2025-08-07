import numpy as np
from pymoo.visualization.scatter import Scatter

F1 = np.loadtxt("resFdf1.txt")
ideal_point = np.min(F1, axis=0)
nadir_point = np.max(F1, axis=0)
F1 = (F1 - ideal_point) / (nadir_point - ideal_point)
F2 = np.loadtxt("df1v5.txt")


plot = Scatter(tight_layout=True, legend=True) 
plot.add(F1, s=10, label="resF")
plot.add(F2, s=10, color='red', label="addF")
plot.show()

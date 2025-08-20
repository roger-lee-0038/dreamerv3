import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.problems.dynamic.df import DF1

F1 = np.loadtxt("resFdf1.txt")
ideal_point = np.min(F1, axis=0)
nadir_point = np.max(F1, axis=0)
#F1 = (F1 - ideal_point) / (nadir_point - ideal_point)
F2 = np.loadtxt("df1v4.txt")
F2 = F2 * (nadir_point - ideal_point) + ideal_point
F3 = np.loadtxt("resFdf1residual.txt")
ideal_point = np.min(F3, axis=0)
nadir_point = np.max(F3, axis=0)
F4 = np.loadtxt("df1v4residual.txt")
F4 = F4 * (nadir_point - ideal_point) + ideal_point

plot = Scatter(tight_layout=True, legend=True) 
plot.add(F1, s=10, color='gold', label="resFdf1")
plot.add(F2, s=10, color='green', label="addFdf1")
plot.add(F3, s=10, color='blue', label="resFdf1residual")
plot.add(F4, s=10, color='red', label="addFdf1residual")
plot.show()

import numpy as np
from pymoo.visualization.scatter import Scatter

F1 = np.loadtxt("resFdf1.txt")
F2 = np.loadtxt("front_11.txt")
F3 = np.loadtxt("front_21.txt")
F4 = np.loadtxt("front_51.txt")


plot = Scatter(tight_layout=True, legend=True) 
plot.add(F1, s=10, label="front_0")
plot.add(F2, s=10, color='red', label="front_11")
plot.add(F3, s=10, color='green', label="front_21")
plot.add(F4, s=10, color='gold', label="front_51")
plot.show()

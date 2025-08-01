import pickle
import numpy as np
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
import sys

a = np.loadtxt("allFdf1.txt")
print("len ys: {}".format(len(a)))
b = np.loadtxt("allXdf1.txt")
print("len xs: {}".format(len(b)))

front_indice = fast_non_dominated_sort(a)
print("len fronts {}".format(len(front_indice)))
for i in range(len(front_indice)):
    print("len front {}: {}".format(i, len(front_indice[i])))
np.savetxt("front_21.txt", a[front_indice[21]])
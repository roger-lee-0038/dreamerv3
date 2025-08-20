import numpy as np

from pymoo.problems.dynamic.df import DF1
from pymoo.visualization.scatter import Scatter

plot = Scatter(legend=True)
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

for index, t in enumerate(np.linspace(0, 10.0, 5)):
    print("t=", t)
    problem = DF1(time=t)
    plot.add(problem.pareto_front(), plot_type="line", color=colors[index], alpha=0.7, label=f"t={t}")

plot.show()

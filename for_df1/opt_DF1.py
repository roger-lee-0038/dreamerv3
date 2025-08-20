import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from pymoo.problems.dynamic.df import DF1
import sys
sys.path.append("..")
from utils import Func, run_funcs

import pickle

if __name__ == '__main__':

    problem = DF1(t=0)

    # create the algorithm object
    algorithm = NSGA2(pop_size=20)

    # let the algorithm object never terminate and let the loop control it
    termination = NoTermination()

    # create an algorithm object that never terminates
    algorithm.setup(problem, termination=termination)

    # fix the random seed manually
    np.random.seed(1)

    # until the algorithm has no terminated
    all_X = []
    all_F = []
    hist_F = []
    totCnt = 0
    for n_gen in range(40):
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()

        # get the design space values of the algorithm
        X = pop.get("X")

        # implement your evluation
        moo_chunk = run_funcs(
            Func(problem.evaluate, 0),
            X,
            4,
        )

        all_X.extend(X)
        all_F.extend([moo for moo in moo_chunk])
        F = np.array([moo for moo in moo_chunk])
        #G = np.array([moo["cons"] for moo in moo_chunk])

        #static = StaticProblem(problem, F=F, G=G)
        static = StaticProblem(problem, F=F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(algorithm.n_gen)
        hist_F.append(algorithm.opt.get("F"))

    # obtain the result objective from the algorithm
    res = algorithm.result()
    np.savetxt("resXdf1.txt", res.X)
    np.savetxt("resFdf1.txt", res.F)
    np.savetxt("allXdf1.txt", np.array(all_X))
    np.savetxt("allFdf1.txt", np.array(all_F))
    with open("histFdf1.pkl", "wb") as handler:
        pickle.dump(hist_F, handler)
    #np.savetxt("resPopXgbo.txt", res.pop.get('X'))
    #np.savetxt("resPopFgbo.txt", res.pop.get('F'))

    # calculate a hash to show that all executions end with the same result
    print("hash", res.F.sum())

    from pymoo.visualization.scatter import Scatter
    plot = Scatter(tight_layout=True)
    plot.add(res.F, s=10)
    #plot.add(res.F[0], s=30, color="red")
    plot.show()

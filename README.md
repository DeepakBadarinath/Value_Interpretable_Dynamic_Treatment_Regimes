# Value_Interpretable_Dynamic_Treatment_Regimes
Given modelling parameters for a Markov Decision Process, we present a greedy method to obtain intepretable tree based policies. We observe that the state space is given by a disjoint union of boxes for which we create a data structure to store. Under appropriate tightness conditions we obtain an optimization of the algorithm. 

We also present ideas of scaling this approach under tightness assumptions. In order to scale the approach, we observe that we must solve the convex envelope problem in D-space. We come with an analytic linear algebraic way to solve the 1D convex envelope problem. 

On the codebase side, we present a multithreaded parallelizable versions of each algorithm suitable for running high dimensional simulated and real world data experiments.

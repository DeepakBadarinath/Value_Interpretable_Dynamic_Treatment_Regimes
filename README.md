# Value_Interpretable_Dynamic_Treatment_Regimes
Given modelling parameters for a Markov Decision Process, we present a greedy method to obtain intepretable tree based policies. We observe that the state space is given by a disjoint union of boxes for which we create a data structure to store. Under appropriate tightness conditions we obtain an optimization of the algorithm. 

We also present ideas of scaling this approach under tightness assumptions. In order to scale the approach, we observe that we must solve the convex envelope problem in D-space. We come with an analytic linear algebraic way to solve the 1D convex envelope problem. Further we extend Borkar's Q-learning approach to high dimensional space and derive error bounds for Borkar's approach and it's extension.

To be done:
1. Scaled version of the simulated experiments
2. Real world data approaches
3. Extension of the theory and experiments for the Tight Value Interpretable Dynamic Regimes approach.

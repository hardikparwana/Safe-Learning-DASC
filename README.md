# Safe-Learning-DASC

This repository implements our ICRA 2022 submission on 

**Recursive Feasibility Guided Optimal Parameter Adaptation of Differential Convex Optimization Policies for Safety-Critical Systems**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

## Description
We pose the question: 

What is the best we can do *given a parametric controller*? How to adapt their parameters in face of different external conditions? And what if the controller is an optimization problem itself? Can we maintain *feasibility over a horizon* if we change the parameter? In other words, how do we relate state-feedback controllers(that depend on current state only) to their long-term performance.

We propose a novel combination of backpropagation for dynamical systems and use FSQP algorithms to update parameters of QP so that:
1. Performance is improved with guarantees of feasible trajectory over the same time horizon.
2. The horizon over which the QP controller remains feasible is increased compared to its previous value.

Adaptive Parameter

video: https://www.youtube.com/embed/WIWQghdr8pQ


https://user-images.githubusercontent.com/19849515/134233753-a6f3fa43-0071-4eef-933c-9a4650e28c48.mp4



https://user-images.githubusercontent.com/19849515/134233782-c9c1c02f-bac2-42ac-a714-5587d883fbdb.mp4


https://user-images.githubusercontent.com/19849515/134233786-ac48c01b-c111-4f07-a0ec-06fda23cf456.mp4



![](https://user-images.githubusercontent.com/19849515/134233753-a6f3fa43-0071-4eef-933c-9a4650e28c48.mp4)

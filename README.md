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


| Adaptive | Constant | Reward Plot |
| -------- | -------- | ----------- |
| <img src="https://user-images.githubusercontent.com/19849515/134234311-9fc31797-b721-4457-9415-a7189ca9b247.gif" width="330" alt="Adaptive" title="hello" /> | <img src="https://user-images.githubusercontent.com/19849515/134234319-a9864ba6-277d-4ca4-a500-4597f596d805.gif" width="330"/> | <img src="https://user-images.githubusercontent.com/19849515/134234324-38a3c582-4c73-422b-8d56-bd31e0229648.gif" width="330"/> |

![](https://user-images.githubusercontent.com/19849515/134234311-9fc31797-b721-4457-9415-a7189ca9b247.gif "Proposed Adaptive Parameter")


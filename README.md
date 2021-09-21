# Safe-Learning-DASC

This repository implements our ICRA 2022 submission on 

**Recursive Feasibility Guided Optimal Parameter Adaptation of Differential Convex Optimization Policies for Safety-Critical Systems**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

## Description
We pose the question: 

What is the best we can do *given a parametric controller*? How to adapt their parameters in face of different external conditions? And what if the controller is an optimization problem itself? Can we maintain *feasibility over a horizon* if we change the parameter? In other words, how do we relate (optimization based) state-feedback controllers(that depend on current state only) to their long-term performance.

![qp_controller](https://user-images.githubusercontent.com/19849515/134256967-0d5b1f4b-aed0-45e7-9f2e-5376c2b713ca.png)


We propose a novel combination of backpropagation for dynamical systems and use FSQP algorithms to update parameters of QP so that:
1. Performance is improved with guarantees of feasible trajectory over the same time horizon.
2. The horizon over which the QP controller remains feasible is increased compared to its previous value.

# Dependencies
The code was run on Ubuntu 20 with Python 3.6 and following packages
- cvxpy==1.1.14
- cvxpylayers==0.1.5
- torch==1.9.0
- matplotlib==3.3.4
- numpy==1.19.5

To run the code, run `source export_setup.sh` from main folder

# 1D Autonomous Car
To examine how the car behaves with different values of parameters, see our notebook example **car_example.ipynb**.

For running our algorithm to increase horizon for feasibility of QP controller, run the following code where you should be able to see improvement of time at which QP fails.
```
python QPpolicy/QPpolicyTorchCar.py 
```


Following figures show how parameters change with each step of proposed GD

| a = 2.5, b=4.5 | a = 1.0, b = 3.0 | a = 3.0, b = 1.0 |
| --------------| -------------------| -----------------|
| ![param1](https://user-images.githubusercontent.com/19849515/134238619-0b8f2729-0f02-479b-b744-f8030934fa20.gif) | ![param2](https://user-images.githubusercontent.com/19849515/134238621-7be16c78-0188-4bbe-94f7-988372a3eb84.gif) | ![param3](https://user-images.githubusercontent.com/19849515/134238629-f443d52b-5f0f-4861-ac36-2c0aa4046fa0.gif) |


![paper_car](https://user-images.githubusercontent.com/19849515/134240335-acbbbf98-9f4f-4da5-a198-accfb27d0a72.png)


# Unicycle Follower
Run the following code
```
python QPpolicy/QPpolicy.py
```


| Adaptive Parameters (proposed) | Constant Parameter | Reward Plot |
| -------- | -------- | ----------- |
| <img src="https://user-images.githubusercontent.com/19849515/134234311-9fc31797-b721-4457-9415-a7189ca9b247.gif" width="300" /> | <img src="https://user-images.githubusercontent.com/19849515/134234319-a9864ba6-277d-4ca4-a500-4597f596d805.gif" width="300"/> | <img src="https://user-images.githubusercontent.com/19849515/134234324-38a3c582-4c73-422b-8d56-bd31e0229648.gif" width="300"/> |


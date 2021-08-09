from cvxpy.problems.objective import Objective
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import os
import time
from numpy.core.numeric import _moveaxis_dispatcher

import matplotlib.cm as cm
import matplotlib.animation as animation

import cvxpy as cp
import torch
from torch.cuda import max_memory_cached

from cvxpylayers.torch import CvxpyLayer
from matplotlib.animation import FFMpegWriter

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from utils.utils import *
from environment_models.obstacles import *

import math

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Sensing Range
FoV = 60*np.pi/180
max_D = 5
min_D = 0.3


# Simulation 
n_robots = 3
n_obstacles = 6
states_per_robot = 2
inputs_per_robot = 2
distance_to_target = 0.5
dt = 0.1

# Map
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-10,10),ylim=(-10,10))



def run(args):

    # Robots
    robots = []
    robots.append( SingleIntegrator(np.array([0,0]),dt,ax,0) )
    robots.append( SingleIntegrator(np.array([1,0]),dt,ax,1) )
    robots.append( SingleIntegrator(np.array([0.5,1]),dt,ax,2) )
    robots.append( SingleIntegrator(np.array([0.5,-1]),dt,ax,3) )

    #Goal Location
    goal_location  = SingleIntegrator(np.array([9,5]),dt,ax,0)
    ax.plot(goal_location.X[0,0],goal_location.X[1,0],'x')

    # plt.show()

    # Environment obstacles
    # obs = rectangle(0,0,1,1,ax,0)
    # obs2 = circle(2,0,1,ax,1)
    obstacles = []
    obstacles.append( circle( 4,0,min_D,ax,0 ) )
    obstacles.append( circle( 6,0,min_D,ax,1 ) )
    obstacles.append( circle( 8,0,min_D,ax,2 ) )
    obstacles.append( circle( 4,3,min_D,ax,3 ) )
    obstacles.append( circle( 6,3,min_D,ax,4 ) )
    obstacles.append( circle( 8,3,min_D,ax,5 ) )

    n_robots = len(robots)
    n_obstacles = len(obstacles)

    for st in range(args.max_steps):

        n_robots = len(robots)

        u = cp.Variable(((n_robots-1)*inputs_per_robot,1))
        u_leader = cp.Variable((inputs_per_robot,1))
        delta = cp.Variable(n_robots)
        ks = cp.Parameter(n_robots,value=args.k*np.ones(n_robots))
        alpha_agents = cp.Parameter((n_robots,n_robots),value=args.alpha_agents*np.ones((n_robots,n_robots)))
        alpha_obs = cp.Parameter((n_robots,n_obstacles),value=args.alpha_obs*np.ones((n_robots,n_obstacles)))
        alpha_con = cp.Parameter(n_robots-1,value=args.alpha_con*np.ones(n_robots-1))
        cons = []
        cons_leader = []

        u_ref = np.zeros((inputs_per_robot,1)) # move straight for leader

        # n_robots = 1

        # Leader Control
        # add barrier constraints to obstacles
        for j in range(n_obstacles):
            gamma_i = alpha_obs[ 0,j ]
            h, dh_dxi = robots[0].obsBarrier(obstacles[j],min_D)
            cons_leader += [ dh_dxi @ robots[0].xdot(u_leader) >= -gamma_i*h + 0.2]

        # Leader desired goal location
        V, dV_dxi, dV_dxj = robots[0].Lyapunov(goal_location,0)
        cons_leader += [ dV_dxi @ robots[0].xdot(u_leader) <= -ks[0] * V + delta[0] ]

        # Objective Function
        u_ref[0,0] = 2.0
        objective_leader = cp.Minimize( cp.sum_squares(u_leader-u_ref) + 10*cp.sum_squares(delta[0]) )
        problem_leader = cp.Problem(objective_leader,cons_leader)

        problem_leader.solve()
        robots[0].step(u_leader.value[0],u_leader.value[1])
        robots[0].render_plot()

        # print("input leader",u_leader.value)
        # print("inputs",u.value)

        # add barrier constraints to agents: Not for Leader
        for i in range(n_robots):
            for j in range(i+1,n_robots):
                if i==0:
                    u_i = u_leader.value
                else:
                    u_i = u[(i-1)*states_per_robot:(i-1)*inputs_per_robot + inputs_per_robot,0]
                u_j = u[(j-1)*states_per_robot:(j-1)*inputs_per_robot + inputs_per_robot,0]
                alpha_ij = alpha_agents[ i,j ]
                alpha_ji = alpha_agents[ j,i ]
                h, dh_dxi, dh_dxj = robots[i].agentBarrier(robots[j],min_D)
                cons += [ dh_dxi @ robots[i].xdot(u_i) + dh_dxj @ robots[j].xdot(u_j) >= -alpha_ij*h]
                cons += [alpha_ij == alpha_ji]


        # add barrier constraints to obstacles
        for i in range(1,n_robots):
            for j in range(i,n_obstacles):
                u_i = u[(i-1)*states_per_robot:(i-1)*inputs_per_robot + inputs_per_robot,0]
                gamma_i = alpha_obs[ i,j ]
                h, dh_dxi = robots[i].obsBarrier(obstacles[j],min_D)
                cons += [ dh_dxi @ robots[i].xdot(u_i) >= -gamma_i*h + 0.2]

        # desired distance and connectivity constraint with Leader
        for i in range(1,n_robots):
            u_i = u[(i-1)*states_per_robot:(i-1)*inputs_per_robot + inputs_per_robot,0]
            u_L = u_leader.value

            # connectivity constraint
            h, dx_dxi, dh_dxj = robots[i].connectivityBarrier(robots[0],max_D)
            cons += [dh_dxi @ robots[i].xdot(u_i) + dh_dxj @ robots[0].xdot(u_L) >= - alpha_con[i-1]*h ]

            # desired distance
            V, dV_dxi, dV_dxj = robots[i].Lyapunov(robots[0],distance_to_target)
            cons += [ dV_dxi @ robots[i].xdot(u_i) <= - ks[i]*V + delta[i]]
            cons += [delta[i-1] >= 0]
            # cons += [cp.abs(u_i[0]) <= 40]
            # cons += [cp.abs(u_i[1]) <= 40]

        

        # Objective Function
        # u_ref[0,0] = 2.0
        objective = cp.Minimize( cp.sum_squares(u) + 100*cp.sum_squares(delta) )

        problem = cp.Problem(objective,cons)
        # assert problem.is_dpp()
        problem.solve(verbose=True)
        # print("status",problem.status)
        if problem.status != 'optimal':
            print("PROBLEM NOT FEASIBLE")
            exit()

        # print("input",u.value[0:2])
        print("delta",delta.value)
        # propagate state
        for i in range(1,n_robots):
            u_i = u.value[(i-1)*states_per_robot:(i-1)*inputs_per_robot + inputs_per_robot,0]
            robots[i].step(u_i[0],u_i[1])
            robots[i].render_plot()

        # for i in range(n_robots):
        #     for j in range(i,n_obstacles):
        #         u_i = u[i*states_per_robot:i*inputs_per_robot + inputs_per_robot,0]
        #         gamma_i = alpha_obs[ i,j ]
        #         h, dh_dxi = robots[i].obsBarrier(obstacles[j],min_D)
        #         # print(i,j,obstacles[j].id,h)
        #         if h<0:
        #             print("************ERROR****************")
        #             exit()

        fig.canvas.draw()
        fig.canvas.flush_events()

        # print("done")


import argparse
parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--max-steps', type=int, default=200, metavar='N',help='maximum number of steps of each episode') #70
parser.add_argument('--alpha_agents', type=float, default=30.0, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha_obs', type=float, default=10.0, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha_con', type=float, default=30.0, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--k', type=float, default=0.5, metavar='G',help='CLF parameter')  #0.003
args = parser.parse_args("")

run(args)

        









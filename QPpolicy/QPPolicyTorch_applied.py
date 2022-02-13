import numpy as np 
import time

import argparse
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from dynamics import *
import traceback

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

class MPC_dynamics:

	@staticmethod
	def forward(ctx, input_x, input_leader):
		'''
			input_x: (n_agent x 1) agent initial state
			input_leader: (n_leader x T) leader trajectpry over next T time steps
		'''

		x = [input_x]

		# N time steps. hard coded

		for i in range(N):
			u = qp_policy(x[-1],input_leader[t])
			x_ = dynam(x[-1],u)
			x.append(input_x)  # correct.. has to be torch append only


		# save tensors for use in backward computation later on
        ctx.save_for_backward(x,u, gx, df_dx, dgxu_dx)
        # print(f"fx:{fx}, gx:{gx}, u:{u}")
        return fx + torch.matmul(gx, u)



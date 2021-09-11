import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from dynamics import *

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

class Actor:
    def __init__(self,alpha=0.1,k=0.1, beta = 0.1, umax=np.array([3,3]),umin=np.array([-3,-3]),horizon = 10) :
        self.alpha_nominal = alpha #1.0 does not work
        self.alpha1_nominal = alpha
        self.alpha2_nominal = alpha
        self.alpha3_nominal = alpha
        self.k_nominal = k
        self.max_action = umax
        self.min_action = umin
        self.horizon = horizon
        self.beta = beta

        self.alpha1_tch = torch.tensor(self.alpha1_nominal, requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor(self.alpha2_nominal, requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor(self.alpha3_nominal, requires_grad=True, dtype=torch.float)
        self.k_tch  = torch.tensor(self.k_nominal, requires_grad=True, dtype=torch.float)

        self.ud_tch = []
        self.ud_nominal = []
        for _ in range(horizon):
            self.ud_tch.append( torch.tensor(np.array([[0],[0]]), requires_grad=True, dtype=torch.float) )
        self.alpha1_grad = torch.tensor([0.0],dtype=torch.float)
        self.alpha2_grad = torch.tensor([0.0],dtype=torch.float)
        self.alpha3_grad = torch.tensor([0.0],dtype=torch.float)
        self.k_grad = torch.tensor([0.0],dtype=torch.float)

    def setParams(self):
        self.alpha1_tch = torch.tensor(self.alpha1_nominal, requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor(self.alpha2_nominal, requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor(self.alpha3_nominal, requires_grad=True, dtype=torch.float)
        self.k_tch  = torch.tensor(self.k_nominal, requires_grad=True, dtype=torch.float)
        for i in range(self.horizon):
            self.ud_tch[i] = torch.tensor(self.ud_nominal[i], requires_grad=True, dtype=torch.float)

    def policy(self,agent,target,horizon_step):

        if len(self.ud_nominal)<self.horizon:
            u, w = agent.nominal_controller(target)
            U = np.array([u,w]).reshape(-1,1)
            self.ud_nominal.append(U)
            self.ud_tch[horizon_step] = torch.tensor(U, requires_grad=True, dtype=torch.float)           
        

        h1,dh1_dxA,dh1_dxB = agent.CBF1_loss(target)
        h2,dh2_dxA,dh2_dxB = agent.CBF2_loss(target)
        h3,dh3_dxA,dh3_dxB = agent.CBF3_loss(target)

        V,dV_dxA,dV_dxB = agent.CLF_loss(target)

        x = cp.Variable((2,1))
        delta = cp.Variable(1)

        alpha1 = cp.Parameter(value=self.alpha1_nominal)
        alpha2 = cp.Parameter(value=self.alpha2_nominal)
        alpha3 = cp.Parameter(value=self.alpha3_nominal)
        k = cp.Parameter(value=self.k_nominal)
        U_d = cp.Parameter((2,1),value=U)

        # QP objective
        objective = cp.Minimize(cp.sum_squares(x-U_d) + 100*delta)
        
        # CLF constraint
        const = [dV_dxA @ agent.xdot(x) + dV_dxB @ target.xdot(target.U) <= -k*V + delta]
        const += [delta>=0]

        # CBF constraints
        const += [dh1_dxA @ agent.xdot(x) + dh1_dxB @ target.xdot(target.U) >= -alpha1*h1 ]
        const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha2*h2 ]
        const += [dh3_dxA @ agent.xdot(x) + dh3_dxB @ target.xdot(target.U) >= -alpha3*h3 ]
        # const += [alpha >= -20]
        problem = cp.Problem(objective,const)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[U_d, alpha1, alpha2, alpha3, k], variables=[x])

        solver_args = {
            'verbose': False,
            'max_iters': 1000000
        }

        # Solve the QP
        #  result = problem.solve()
        try:
            solution, = cvxpylayer(self.ud_tch[horizon_step], self.alpha1_tch, self.alpha2_tch, self.alpha3_tch, self.k_tch, solver_args=solver_args)
        except:
            print("SBF QP not solvable")
            return False, np.array([0,0]).reshape(-1,1), 0, 0

        return solution  # A tensor

    def updateParameters(self,rewards):
        policy_gradient = []

        objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
        for index, value in enumerate(rewards): #index from 0 to horizon -1
            # print("obj", objective_tensor)
            objective_tensor = objective_tensor + value
            

        # print("Objective Tensor", objective_tensor)
        objective_tensor.backward(retain_graph=True)

        # Get Gradients
        alpha1_grad = self.alpha1_tch.grad - self.alpha1_grad
        alpha2_grad = self.alpha2_tch.grad - self.alpha2_grad
        alpha3_grad = self.alpha3_tch.grad - self.alpha3_grad
        k_grad = self.k_tch.grad #- self.k_grad

        self.alpha1_grad = self.alpha1_grad + alpha1_grad
        self.alpha2_grad = self.alpha2_grad + alpha2_grad
        self.alpha3_grad = self.alpha3_grad + alpha3_grad
        self.k_grad = self.k_tch.grad + k_grad

        print(objective_tensor, alpha1_grad, alpha2_grad, alpha3_grad, k_grad)
        ud_grad = []
        for i in range(self.horizon):
            ud_grad.append( self.ud_tch[i].grad  )

        ## TODO: write code for constrained GD here
        self.alpha1_nominal = self.alpha1_nominal + self.beta*alpha1_grad.detach().numpy()
        self.alpha2_nominal = self.alpha2_nominal + self.beta*alpha2_grad.detach().numpy()
        self.alpha3_nominal = self.alpha3_nominal + self.beta*alpha3_grad.detach().numpy()
        self.k_nominal = self.k_nominal + self.beta*k_grad.detach().numpy()

        # clipping > 0
        if self.alpha1_nominal < 0:
            self.alpha1_nominal = 0
        if self.alpha2_nominal < 0:
            self.alpha2_nominal = 0
        if self.alpha3_nominal < 0:
            self.alpha3_nominal = 0
        if self.k_nominal < 0:
            self.k_nominal = 0
        # print(f"alpha1_nom:{self.actor.alpha_nominal}, alpha2_nom:{self.actor.alpha2_nominal}, alpha3_nom:{self.actor.alpha3_nominal} k_nominal:{self.actor.k_nominal}")

def train(args):

    # Visualization setup
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(xlim=(0,10),ylim=(-5,5))
    lines, = ax.plot([],[],'o-')
    areas, = ax.fill([],[],'r',alpha=0.1)
    bodyF = ax.scatter([],[],c='r',s=10)            
    bodyF_tensor = ax.scatter([],[],c='c',s=5)      
    bodyT = ax.scatter([],[],c='g',s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    # set robots
    t = 0
    dt = args.dt
    agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
    agentF.d1 = 0
    agentF.d2 = 0
    agentF.d3 = 0
    agentT = SingleIntegrator(np.array([1,0]),dt,ax,0)

    #Set policy agent
    actor = Actor(alpha=args.alpha,k=args.k, horizon=args.horizon, beta=args.lr_actor)

    # Initialize tensor list
    state_tensors = [torch.tensor(agentF.X,dtype=torch.float,requires_grad=True)]
    input_tensors = []

    dynam = torch_dynamics.apply
    rewards = []

    # Update to moving horizon
    for horizon_step in range(args.horizon):


        # print("iter",horizon_step)

        uL = 0.5
        vL = 2.6*np.sin(np.pi*t) #  0.1 # 1.2

        u = actor.policy(agentF,agentT,horizon_step)  # tensor    
        x = state_tensors[-1]
        
        x_ = dynam(x,u)       
        
        # compute reward: should be a reward
        # r = compute_reward(x_)
        
        # Store state and input tensors
        state_tensors.append(x_)
        input_tensors.append(u)
        rewards.append( agentF.compute_reward_tensor(x_,agentT) )
        print(rewards[-1])
        
        # visualize
        FX = agentF.step(u.detach().numpy())  
        agentT.step(uL,vL) 

        # animation plot
        lines, areas, bodyF = agentF.render(lines,areas,bodyF)
        bodyT = agentT.render(bodyT)
        bodyF_tensor.set_offsets([x_.detach().numpy()[0,0],x_.detach().numpy()[1,0]])
        
        fig.canvas.draw()
        fig.canvas.flush_events()

        t += dt

    print("Update 1")
    actor.updateParameters(rewards)
    print("Update 2")
    actor.updateParameters(rewards)
    print("Update 3")
    actor.updateParameters(rewards)




























parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="UnicycleFollower")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--lr_actor', type=float, default=0.03, metavar='G',help='learning rate of actor')  #0.003
parser.add_argument('--lr-critic', type=float, default=0.03, metavar='G',help='learning rate of critic') #0.003
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='batch size (default: 256)') #100
parser.add_argument('--buffer-capacity', type=int, default=20, metavar='N', help='buffer_capacity') #10
parser.add_argument('--max-steps', type=int, default=200, metavar='N',help='maximum number of steps of each episode') #70
parser.add_argument('--total-episodes', type=int, default=1, metavar='N',help='total training episodes') #1000
parser.add_argument('--policy-freq', type=int, default=500, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
parser.add_argument('--horizon', type=int, default=10, metavar='N',help='RL time horizon') #3
parser.add_argument('--alpha', type=float, default=0.15, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--train', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie_name', default="test.mp4")
parser.add_argument('--dt', type=float, default="0.01")
args = parser.parse_args("")


train(args)
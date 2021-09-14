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

    def policy(self,X,agent,target,horizon_step):

        if len(self.ud_nominal)<self.horizon:
            # print("Nominal")
            u, w = agent.nominal_controller(target)
            U = np.array([u,w]).reshape(-1,1)
            self.ud_nominal.append(U)
            self.ud_tch[horizon_step] = torch.tensor(U, requires_grad=True, dtype=torch.float)           
        else:
            u, w = agent.nominal_controller(target)
            U = np.array([u,w]).reshape(-1,1)

        V_,dV_dxA_f_, dV_dxA_g_,dV_dxB_ = agent.CLF_loss_tensor(X,target)
        h1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_ = agent.CBF1_loss_tensor(X,target)
        h2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_ = agent.CBF2_loss_tensor(X,target)
        h3_,dh3_dxA_f_, dh3_dxA_g_,dh3_dxB_ = agent.CBF3_loss_tensor(X,target)
        ah1_ = self.alpha1_tch*h1_  
        ah2_ = self.alpha2_tch*h2_  
        ah3_ = self.alpha3_tch*h3_  
        # print("ah1", ah1_)
        kV_ = self.k_tch * V_

        u = cp.Variable((2,1))
        delta = cp.Variable(1)

        ah1 = cp.Parameter(1)
        dh1_dxA_f = cp.Parameter(1)
        dh1_dxA_g = cp.Parameter((1,2))
        dh1_dxB = cp.Parameter((1,2))

        ah2 = cp.Parameter(1)
        dh2_dxA_f = cp.Parameter(1)
        dh2_dxA_g = cp.Parameter((1,2))
        dh2_dxB = cp.Parameter((1,2))

        ah3 = cp.Parameter(1)
        dh3_dxA_f = cp.Parameter(1)
        dh3_dxA_g = cp.Parameter((1,2))
        dh3_dxB = cp.Parameter((1,2))

        kV = cp.Parameter(1)
        dV_dxA_f = cp.Parameter(1)
        dV_dxA_g = cp.Parameter((1,2))
        dV_dxB = cp.Parameter((1,2))

        U_d = cp.Parameter((2,1),value=U)
        x = cp.Parameter((3,1))

        # QP objective
        objective = cp.Minimize(cp.sum_squares(u-U_d) + 100*delta)
        
        # CLF constraint
        const = [dV_dxA_f + dV_dxA_g @ u + dV_dxB @ target.xdot(target.U) <= -kV + delta]
        const += [delta>=0]

        # print("h1")
        # print(dh1_dxA_f_, dh1_dxA_g_, dh1_dxB_, h1_, ah1_, X, target.X)
        # print("h2")
        # print(dh2_dxA_f_, dh2_dxA_g_, dh2_dxB_, h2_, ah2_, X, target.X)
        # print("h3")
        # print(dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_, h3_, ah3_, X, target.X)
        # print("V")
        # print(dV_dxA_f_, dV_dxA_g_, dV_dxB_, V_, kV_, X, target.X)
        # print("k",self.k_tch)

        # CBF constraints
        const += [dh1_dxA_f + dh1_dxA_g @ u + dh1_dxB @ target.xdot(target.U) >= -ah1 ]
        const += [dh2_dxA_f + dh2_dxA_g @ u + dh2_dxB @ target.xdot(target.U) >= -ah2 ]
        const += [dh3_dxA_f + dh3_dxA_g @ u + dh3_dxB @ target.xdot(target.U) >= -ah3 ]
        # const += [alpha >= -20]
        problem = cp.Problem(objective,const)
        assert problem.is_dpp()
        # exit()

        cvxpylayer = CvxpyLayer(problem, parameters=[U_d, ah1,dh1_dxA_f, dh1_dxA_g,dh1_dxB, ah2,dh2_dxA_f, dh2_dxA_g,dh2_dxB, ah3,dh3_dxA_f,dh3_dxA_g,dh3_dxB,  kV,dV_dxA_f,dV_dxA_g,dV_dxB  ], variables=[u])

        solver_args = {
            'verbose': False,
            'max_iters': 1000000
        }

        # Solve the QP
        #  result = problem.solve()
        # print("ud",self.ud_tch[horizon_step])
        # print(ah1_.shape)
        # print(ah1.shape)
        try:
            solution, = cvxpylayer(self.ud_tch[horizon_step], ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_, solver_args=solver_args)
        except:
            print("SBF QP not solvable")
            traceback.print_exc()
            return False, np.array([0,0]).reshape(-1,1), 0, 0
        # exit()
        # print("solution",solution)
        return solution[0]  # A tensor

    def updateParameters(self,rewards):
        policy_gradient = []

        objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
        for index, value in enumerate(rewards): #index from 0 to horizon -1
            # print("obj", objective_tensor)
            objective_tensor = objective_tensor + value
            

        # print("Objective Tensor", objective_tensor)
        objective_tensor.backward(retain_graph=True)

        # Get Gradients
        print("g2",self.alpha1_tch.grad)
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
    dynam = torch_dynamics.apply

    for ep in range(args.total_episodes):

        # Initialize tensor list
        state_tensors = [torch.tensor(agentF.X,dtype=torch.float,requires_grad=True)]
        input_tensors = []
        rewards = []

        # Rollout: Update to moving horizon
        t_roll = t
        for horizon_step in range(args.horizon):
    

            # print("iter",horizon_step)        

            uL = 0.5
            vL = 2.6*np.sin(np.pi*t_roll) #  0.1 # 1.2

            x = state_tensors[-1]
            u = actor.policy(x, agentF,agentT,horizon_step)  # tensor    
            # print("u",u)
            x_ = dynam(x,u)  
            # print("x_",x_)     
            
            # compute reward: should be a reward
            # r = compute_reward(x_)
            
            # Store state and input tensors
            state_tensors.append(x_)
            input_tensors.append(u)
            rewards.append( agentF.compute_reward_tensor(x_,agentT) )

            t_roll += dt

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
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
from numpy.core.fromnumeric import trace

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator1D import *
from dynamics1D import *
import traceback

from torchviz import make_dot

import warnings

warnings.filterwarnings('ignore')

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

class Actor:
    def __init__(self,alpha=0.1, alpha1 = 0.1, alpha2 = 0.1, k=0.1, beta = 0.1, umax=np.array([3,3]),umin=np.array([-3,-3]),horizon = 10) :
        self.alpha_nominal = alpha #1.0 does not work
        self.alpha1_nominal = alpha1
        self.alpha2_nominal = alpha2
        self.k_nominal = k
        self.max_action = umax
        self.min_action = umin
        self.horizon = horizon
        self.beta = beta

        self.alpha1_tch = [torch.tensor([self.alpha1_nominal], requires_grad=True, dtype=torch.float)]
        self.alpha2_tch = [torch.tensor([self.alpha2_nominal], requires_grad=True, dtype=torch.float)]
      
        self.index_tensors = []
        self.index_tensor_grads = []

        self.alpha1_grad_sum = 0
        self.alpha2_grad_sum = 0

    def resetParams(self):
        self.alpha1_tch = [torch.tensor([self.alpha1_nominal], requires_grad=True, dtype=torch.float)]
        self.alpha2_tch = [torch.tensor([self.alpha2_nominal], requires_grad=True, dtype=torch.float)]
        self.index_tensors = []
        self.index_tensor_grads = []
        self.alpha1_grad_sum = 0
        self.alpha2_grad_sum = 0
        

    # @ignore_warnings(category=UserWarning)
    def policy(self,X,agent,horizon_step, t, c=0.3):

        # t = torch.tensor(t,dtype=torch.float)

        x = cp.Parameter(1,value=X.detach().numpy())
        u = cp.Variable(1) 
        alpha1_v = cp.Variable(1) # for alpha1
        alpha2_v = cp.Variable(1) # for alpha2

        alpha1 = cp.Parameter(1,value = [self.alpha1_nominal])
        alpha2 = cp.Parameter(1,value = [self.alpha2_nominal])

        h1 = x - t
        h1_dot = u - 1

        h2 = 1 + c*t - x 
        h2_dot = c - u

        const =  [h1_dot >= -alpha1_v*h1]
        const += [alpha1_v==alpha1]

        const +=  [h2_dot >= -alpha2_v*h2]
        const += [alpha2_v==alpha2]

        # QP objective
        objective = cp.Maximize(u)
        problem = cp.Problem(objective,const)
        assert problem.is_dpp()


        cvxpylayer = CvxpyLayer(problem, parameters=[x, alpha1, alpha2 ], variables=[u])

        solver_args = {
            'verbose': False,
            'max_iters': 1000000,
        }
        alpha1_tch = self.alpha1_tch[-1] + 0
        alpha2_tch = self.alpha2_tch[-1] + 0
  

        try:
            # print("passed",alpha1_tch)
            solution, = cvxpylayer(X, alpha1_tch, alpha2_tch, solver_args=solver_args)

            # Now get gradients of both the constraints
            h1_ = X - t 
            h1_dot_ = solution - 1 

            h2_ = 1 + c*t - X            
            h2_dot_ = c - solution

            loss1 = h1_dot_ + alpha1_tch*h1_ 
            if loss1.detach().numpy()<-0.01:
                print("ERROR")

            loss1.backward(retain_graph=True)            
            grad1 = [self.alpha1_tch[0].grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad1[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad1[1]
            
            loss2 = h2_dot_ + alpha2_tch*h2_ 
            if loss2.detach().numpy()<-0.01:
                print("ERROR")
                # exit()
            loss2.backward(retain_graph=True)
            grad2 = [self.alpha1_tch[0].grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad2[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad2[1]

            self.index_tensors.append(loss1.detach().numpy()[0])
            self.index_tensors.append(loss2.detach().numpy()[0])
            self.alpha1_tch.append(alpha1_tch)
            self.alpha2_tch.append(alpha2_tch)
            self.index_tensor_grads.append(grad1)
            self.index_tensor_grads.append(grad2)

        except:
            # print("SBF QP not solvable")
            # traceback.print_exc()

            delta1 = cp.Variable(1)
            delta2 = cp.Variable(1)

            const =  [h1_dot >= -alpha1_v*h1 - delta1]
            const += [alpha1_v==alpha1]
            const += [delta1>=0]

            const +=  [h2_dot >= -alpha2_v*h2 - delta2]
            const += [alpha2_v==alpha2]
            const += [delta2>=0]

            # QP objective 1
            objective = cp.Minimize(100000*delta1 + delta2)
            problem = cp.Problem(objective,const)
            problem.solve()

            # print(f"Resolve slacks ", delta1.value, delta2.value)
            if problem.status != 'optimal':
                print("ERROR in Resolve")

            delta2_min = delta2.value

            # QP objective 2
            objective = cp.Minimize(100000*delta2 + delta1)
            problem = cp.Problem(objective,const)
            problem.solve()

            # print(f"Resolve slacks ", delta1.value, delta2.value)
            if problem.status != 'optimal':
                print("ERROR in Resolve")

            delta1_min = delta1.value

            const_index = []
            if delta1.value > 0.0:
                const_index.append(0)
            if delta2.value > 0.0:
                const_index.append(1)


            if delta1_min <= delta2_min:
                const_index = [0]
            else:
                const_index = [1]

            # Now get gradients of both the constraints
            h1_ = X - t 
            h1_dot_ = - 1 

            h2_ = 1 + c*t - X            
            h2_dot_ = c

            loss1 = h1_dot_ + alpha1_tch*h1_ 
            loss1.backward(retain_graph=True)            
            grad1 = [self.alpha1_tch[0].grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad1[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad1[1]
            

            loss2 = h2_dot_ + alpha2_tch*h2_ 
            loss2.backward(retain_graph=True)
            grad2 = [self.alpha1_tch[0].grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad2[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad2[1]

            self.index_tensors.append(loss1.detach().numpy()[0])
            self.index_tensors.append(loss2.detach().numpy()[0])
            self.alpha1_tch.append(alpha1_tch)
            self.alpha2_tch.append(alpha2_tch)
            self.index_tensor_grads.append(grad1)
            self.index_tensor_grads.append(grad2)

            # print(f"grad1:{grad1}, grad2:{grad2}, 1sum:{self.alpha1_grad_sum}, 2sum:{self.alpha2_grad_sum}, delta1_min:{delta1_min}, delta2_min:{delta2_min}")

        
            return False, const_index

        solution.retain_grad()
        return solution, False  # A tensor

    def updateParameters(self,rewards):
        policy_gradient = []

        objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
        for index, value in enumerate(rewards): #index from 0 to horizon -1
            # print("obj", objective_tensor)
            objective_tensor = objective_tensor + value
            
        objective_tensor.backward(retain_graph=True)

        # Get Gradients
        alpha1_grad = self.alpha1_tch.grad - self.alpha1_grad
        alpha2_grad = self.alpha2_tch.grad - self.alpha2_grad

        self.alpha1_grad = self.alpha1_grad + alpha1_grad
        self.alpha2_grad = self.alpha2_grad + alpha2_grad

        print(objective_tensor, alpha1_grad, alpha2_grad)#, alpha3_grad)#, k_grad)
        ud_grad = []
        for i in range(self.horizon):
            ud_grad.append( self.ud_tch[i].grad  )

        self.alpha1_nominal = self.alpha1_nominal + self.beta*alpha1_grad.detach().numpy()
        self.alpha2_nominal = self.alpha2_nominal + self.beta*alpha2_grad.detach().numpy()

        # clipping > 0
        if self.alpha1_nominal < 0:
            self.alpha1_nominal = 0
        if self.alpha2_nominal < 0:
            self.alpha2_nominal = 0

        episode_reward = objective_tensor.detach().numpy()[0]
        return episode_reward
       

    def updateParameterFeasibility(self,rewards,indexes):
        # No need to make new tensors here

        # find feasible direction with QP first
        N = len(self.index_tensors) - 2
        d = cp.Variable((2,1))

        # of feasible time horizon
        e = cp.Parameter((N,1), value = np.asarray(self.index_tensors[0:-2]).reshape(-1,1))
        grad_e = cp.Parameter((N,2), value = np.asarray(self.index_tensor_grads[0:-2]))

        st = e.value>=-0.01
        if [False] in st:
            print (st)

        const = [e + grad_e @ d >= -0.01]
        const += [cp.abs(d[0,0])<= 100]
        const += [cp.abs(d[1,0])<=100]

        ## find direction of final feasibility
        infeasible_constraint = self.index_tensors[-2:][indexes[0]]
        infeasible_constraint_grad = self.index_tensor_grads[-2:][indexes[0]]
        d_infeasible = cp.Parameter((2,1),value = np.asarray(infeasible_constraint_grad).reshape(-1,1))
        objective = cp.Maximize(d.T @ d_infeasible)
        problem = cp.Problem(objective, const)
        problem.solve()
        if problem.status!='optimal':
            return False, 

        self.alpha1_nominal = self.alpha1_nominal + self.beta*d.value[0,0]
        self.alpha2_nominal = self.alpha2_nominal + self.beta*d.value[1,0]

        return True


def train(args):

    # Visualization setup
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(xlim=(0,2),ylim=(-0.5,0.5))
    bodyF = ax.scatter([],[],c='g',s=10)            
    bodyF_tensor = ax.scatter([],[],c='c',s=5)      
    bodyT1 = ax.scatter([],[],c='r',s=10)
    bodyT2 = ax.scatter([],[],c='r',s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    # set robots
    t = 0
    dt = args.dt

    #Set policy agent
    actor = Actor(alpha=args.alpha,alpha1 = args.alpha1, alpha2=args.alpha2, k=args.k, horizon=args.horizon, beta=args.lr_actor)
    dynam = torch_dynamics.apply

    episode_rewards = []

    parameters = []
    parameters.append([args.alpha1, args.alpha2])
    ftime = []

    for ep in range(args.total_episodes):

        # Initialize tensor list
        agentF = SingleIntegrator1D(np.array([0.1]),dt)
        state_tensors = [torch.tensor(agentF.X,dtype=torch.float,requires_grad=True)]
        input_tensors = []
        rewards = []        

        # Rollout: Update to moving horizon
        t_roll = 0# t
        c = 0.3
        for horizon_step in range(args.horizon):    

            # print("iter",horizon_step)        

            uL = 0.5
            vL = 2.6*np.sin(np.pi*t_roll) #  0.1 # 1.2

            x = state_tensors[-1]
            u, indexes = actor.policy(x, agentF,horizon_step, t_roll, c = c)  # tensor    
            if u==False:
                print("QP Failed at ", horizon_step)
                ftime.append(horizon_step)
                break
            x_ = dynam(x,u)  
            x_.retain_grad()
            # print(f"x_:{x_}, u:{u}")
            
            # Store state and input tensors
            state_tensors.append(x_)
            input_tensors.append(u)
            rewards.append( agentF.compute_reward_tensor(x_,t,c) )

            t_roll += dt

             # visualize
            FX = agentF.step(u.detach().numpy())  
            # print(f"xstep:{FX}, xtensor:{x_.detach().numpy()}")

            # # animation plot
            # bodyF = agentF.render(bodyF)
            # bodyT1.set_offsets([t_roll,0.0])
            # bodyT2.set_offsets([1 + c*t_roll,0.0])
            # bodyF_tensor.set_offsets([x_.detach().numpy(),0])
            
            # fig.canvas.draw()
            # fig.canvas.flush_events()

        t += dt
        # print("done")

        if indexes == False:
            print("Parameter Successful!")
            break
        else:
            # Update parameter feasibility
            success = actor.updateParameterFeasibility(rewards, indexes)
            if not success:
                break
            parameters.append([actor.alpha1_nominal, actor.alpha2_nominal])
            actor.resetParams()

    print("Parameters ", parameters)
    print("horizons", ftime)
    



parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="UnicycleFollower")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--lr_actor', type=float, default=0.01, metavar='G',help='learning rate of actor')  #0.003
parser.add_argument('--lr-critic', type=float, default=0.03, metavar='G',help='learning rate of critic') #0.003
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='batch size (default: 256)') #100
parser.add_argument('--buffer-capacity', type=int, default=20, metavar='N', help='buffer_capacity') #10
parser.add_argument('--max-steps', type=int, default=200, metavar='N',help='maximum number of steps of each episode') #70
parser.add_argument('--total-episodes', type=int, default=20, metavar='N',help='total training episodes') #1000
parser.add_argument('--policy-freq', type=int, default=500, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
parser.add_argument('--horizon', type=int, default=500, metavar='N',help='RL time horizon') #3
parser.add_argument('--alpha', type=float, default=3, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha1', type=float, default=0.2, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha2', type=float, default=4.5, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--train', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie_name', default="test.mp4")
parser.add_argument('--dt', type=float, default="0.01")
args = parser.parse_args("")

train(args)

# alpha1 = [3.0, 0.2]
# alpha2 = [1.0, 4.5]
# x0 = [0.5, 0.1]
# beta = [0.01, 0.01]
# times = [[91,91,90,feasible],
#         [62,62,62,62,62]]


## Case 1
# Parameters  [[3.0, 1.0], [2.171817112678339, 1.999999997295347], [3.149193806179822, 0.9999999973420672], [4.149193804906375, 1.9999999973167402], [5.149193804670515, 2.999999997309681], [6.14919380452549, 3.9999999972005984], [7.1491938043940415, 4.9999999965909145], [8.149193804169093, 5.9999999921789255], [9.149193804161591, 6.99999998978608], [10.149193800134931, 7.9999996582562485], [11.149193800057962, 8.999999357340316], [12.149193799161653, 9.999994256402568], [13.149193799160955, 10.99998953919495], [14.149193799157668, 11.999989268083645], [15.149193799155823, 12.999928299894096], [15.146057142741926, 12.78926971117735], [16.146057141765187, 12.17628006653447], [16.876982002935126, 11.17634584747435], [17.876982002934653, 12.176344966640798], [18.615299787980533, 11.176394448885873], [19.615299787980153, 12.176394299293307]]
# horizons [91, 97, 92, 116, 123, 127, 129, 131, 132, 134, 134, 135, 136, 136, 137, 137, 137, 137, 138, 138]


## Case 2
# Parameters  [[0.2, 4.5], [1.1999999999843958, 5.499999997749792], [2.199999999966882, 6.49999999656648], [3.1999993428084377, 6.329111179965787], [4.1999993427987325, 7.329111178175456], [5.199993037924893, 7.147709657262412], [6.199991681206735, 6.967010488329249], [7.199990710225937, 6.787007164091134], [8.199990710126746, 7.787007124088467], [9.199990709824057, 8.78700687050283], [10.199990709490926, 9.786998399741117], [11.199990709489308, 10.78699801326706], [12.199990709488205, 11.78698996358571], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189], [13.199990709487812, 12.786989808532189]]
# horizons [43, 69, 98, 112, 120, 124, 127, 129, 131, 132, 134, 134, 135, 136, 136, 136, 136, 136, 136, 136]
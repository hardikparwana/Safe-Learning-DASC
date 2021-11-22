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
    def __init__(self,alpha=0.1, alpha1 = 0.1, alpha2 = 0.1, alpha3 =0.1, k=0.1, beta = 0.1, umax=np.array([3,3]),umin=np.array([-3,-3]),horizon = 10) :
        self.alpha_nominal = alpha #1.0 does not work
        self.alpha1_nominal = alpha1
        self.alpha2_nominal = alpha2
        self.alpha3_nominal = alpha3
        self.k_nominal = k
        self.ud_nominal = []
        self.max_action = umax
        self.min_action = umin
        self.horizon = horizon
        self.beta = beta

        self.alpha1_tch = torch.tensor([self.alpha1_nominal], requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor([self.alpha2_nominal], requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor([self.alpha3_nominal], requires_grad=True, dtype=torch.float)
        self.k_tch  = torch.tensor(self.k_nominal, requires_grad=True, dtype=torch.float)
        # self.ud_tch = []        

        # for _ in range(horizon):
        #     self.ud_tch.append( torch.tensor(np.array([[0],[0]]), requires_grad=True, dtype=torch.float) )
        
        self.index_tensors = []
        self.index_tensor_grads = []

        self.alpha1_grad_sum = 0
        self.alpha2_grad_sum = 0
        self.alpha3_grad_sum = 0
        self.k_grad_sum = 0
        self.ud_grad_sum = [0,0]

    def resetParams(self):
        self.alpha1_tch = torch.tensor(self.alpha1_nominal, requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor(self.alpha2_nominal, requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor(self.alpha3_nominal, requires_grad=True, dtype=torch.float)
        self.k_tch  = torch.tensor(self.k_nominal, requires_grad=True, dtype=torch.float)
        # for i in range(self.horizon):
        #     self.ud_tch[i] = torch.tensor(self.ud_nominal[i], requires_grad=True, dtype=torch.float)
        self.index_tensors = []
        self.index_tensor_grads = []
        self.alpha1_grad_sum = 0
        self.alpha2_grad_sum = 0
        self.alpha3_grad_sum = 0
        self.k_grad_sum = 0
        self.ud_grad_sum = [0,0]

    def policy(self,X,agent,target,horizon_step):
        
        #agent nominal velocity
        U_d_ = agent.nominal_controller_tensor(X,target)
        # print("U_d",U_d_)
        
        #target tensor
        target_xdot_ = target.xdot(target.U)
        
        # define tensors
        V_,dV_dxA_f_, dV_dxA_g_,dV_dxB_temp = agent.CLF_loss_tensor(X,target)
        h1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_temp = agent.CBF1_loss_tensor(X,target)
        h2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_temp = agent.CBF2_loss_tensor(X,target)
        h3_,dh3_dxA_f_, dh3_dxA_g_,dh3_dxB_temp = agent.CBF3_loss_tensor(X,target)
        dV_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh1_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh2_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh3_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_

        # dh3_dxA_g_.sum().backward(retain_graph=True)
        # alpha1_grad = X.grad #- self.alpha1_grad_sum
        # # self.alpha1_grad_sum += alpha1_grad
        # print("alpha1 grad policy", alpha1_grad)

        ah1_ = self.alpha1_tch*h1_  
        ah2_ = self.alpha2_tch*h2_  
        ah3_ = self.alpha3_tch*h3_  
        # print("ah1", ah1_)
        kV_ = self.k_tch * V_       

        solver_args = {
            'verbose': False,
            'max_iters': 1000000
        }

        try:
            solution, = cvxpylayer(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=solver_args)

            ssum = solution.sum()
            # ssum = self.alpha1_tch[-1].sum()
            ssum.backward(retain_graph=True)
            # grad1_temp = self.alpha1_tch[0].grad.detach().numpy()[0]-self.alpha1_grad_sum
            grad1_temp = self.alpha1_tch.grad.detach().numpy()[0]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad1_temp
            # print("ussum grad 1", grad1_temp)


            # Now get gradients of all the constraints\
            loss1 = dh1_dxA_f_ + torch.matmul(dh1_dxA_g_, solution) + dh1_dxB_target_xdot_ + ah1_
            loss2 = dh2_dxA_f_ + dh2_dxA_g_ @ solution + dh2_dxB_target_xdot_ + ah2_ 
            loss3 = dh3_dxA_f_ + dh3_dxA_g_ @ solution + dh3_dxB_target_xdot_  + ah3_
            loss4 = -dV_dxA_f_ - dV_dxA_g_ @ solution - dV_dxB_target_xdot_ - kV_ #- delta
 
            if loss1.detach().numpy()<-0.01 or loss2.detach().numpy()<-0.01 or loss2.detach().numpy()<-0.01:
                print("ERROR")

            loss1.backward(retain_graph=True)       
            grad1 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad1[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad1[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad1[2]
            self.k_grad_sum = self.k_grad_sum + grad1[3]
            # print("GRAD1",grad1)
            # print(len(self.alpha1_tch))
            # for alphas in self.alpha1_tch:
            #     print("hello")
            #     print(alphas.grad)
            # print(alpha1_tch.grad.detach().numpy(),alpha2_tch.grad.detach().numpy(),alpha3_tch.grad.detach().numpy(),k_tch.grad.detach().numpy())

            # s_time = time.time()
            loss2.backward(retain_graph=True)            
            grad2 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            # print("One time ", time.time()-s_time)
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad2[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad2[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad2[2]
            self.k_grad_sum = self.k_grad_sum + grad2[3]
            # print("GRAD2",grad2)

            loss3.backward(retain_graph=True)            
            grad3 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad3[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad3[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad3[2]
            self.k_grad_sum = self.k_grad_sum + grad3[3]
            # print("GRAD3",grad3)

            loss4.backward(retain_graph=True)            
            grad4 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad4[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad4[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad4[2]
            self.k_grad_sum = self.k_grad_sum + grad4[3]

            self.index_tensors.append(loss1.detach().numpy()[0])
            self.index_tensors.append(loss2.detach().numpy()[0])
            
            self.index_tensors.append(loss3.detach().numpy()[0])
            self.index_tensors.append(loss4.detach().numpy()[0])
            
            self.index_tensor_grads.append(grad1)
            self.index_tensor_grads.append(grad2)
            self.index_tensor_grads.append(grad3)
            self.index_tensor_grads.append(grad4)

        except:
            print("SBF QP not solvable")
            # traceback.print_exc()
            # return False, np.array([0,0]).reshape(-1,1), 0, 0

            delta1 = cp.Variable(1)
            delta2 = cp.Variable(1)
            delta3 = cp.Variable(1)

            # CLF constraint
            const = [dV_dxA_f + dV_dxA_g @ u + dV_dxB_target_xdot  <= -kV + delta]
            const += [delta>=0]

            # CBF constraints
            const += [dh1_dxA_f + dh1_dxA_g @ u + dh1_dxB_target_xdot >= -ah1 - delta1 ]
            const += [dh2_dxA_f + dh2_dxA_g @ u + dh2_dxB_target_xdot >= -ah2 - delta2]
            const += [dh3_dxA_f + dh3_dxA_g @ u + dh3_dxB_target_xdot >= -ah3 - delta3]
            const += [delta1 >= 0]
            const += [delta2 >= 0]
            const += [delta3 >= 0]

            # QP objective 1
            objective = cp.Minimize(100000*delta1 + 100000*delta2 + delta3)
            problem = cp.Problem(objective,const)
            problem.solve()
            # print(f"Resolve slacks ", delta1.value, delta2.value)
            if problem.status != 'optimal':
                print("ERROR in Resolve")
            delta3_min = delta3.value

            # QP objective 2
            objective = cp.Minimize(delta1 + 100000*delta2 + 100000*delta3)
            problem = cp.Problem(objective,const)
            problem.solve()
            # print(f"Resolve slacks ", delta1.value, delta2.value)
            if problem.status != 'optimal':
                print("ERROR in Resolve")
            delta1_min = delta1.value

            # QP objective 3
            objective = cp.Minimize(1000000*delta1 + delta2 + 100000*delta3)
            problem = cp.Problem(objective,const)
            problem.solve()
            # print(f"Resolve slacks ", delta1.value, delta2.value)
            if problem.status != 'optimal':
                print("ERROR in Resolve")
            delta2_min = delta2.value

            const_index = []
            if delta1.value > 0.0:
                const_index.append(0)
            if delta2.value > 0.0:
                const_index.append(1)
            index = np.argmin(np.array([delta1_min, delta2_min, delta3_min]))
            const_index.append(0)

            # Now get gradients of all the constraints
            loss1 = dh1_dxA_f + dh1_dxB_target_xdot + ah1
            loss2 = dh2_dxA_f + dh2_dxB_target_xdot + ah2 
            loss3 = dh3_dxA_f + dh3_dxB_target_xdot + ah3
            loss4 = -dV_dxA_f - dV_dxB_target_xdot - kV

            if loss1.detach().numpy()<-0.01 or loss2.detach().numpy()<-0.01 or loss2.detach().numpy()<-0.01:
                print("ERROR")

            loss1.backward(retain_graph=True)            
            grad1 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad1[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad1[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad1[2]
            self.k_grad_sum = self.k_grad_sum + grad1[3]

            # print(grad1)

            loss2.backward(retain_graph=True)            
            grad2 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad2[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad2[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad2[2]
            self.k_grad_sum = self.k_grad_sum + grad2[3]

            loss3.backward(retain_graph=True)            
            grad3 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad3[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad3[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad3[2]
            self.k_grad_sum = self.k_grad_sum + grad3[3]

            loss4.backward(retain_graph=True)            
            grad4 = [self.alpha1_tch.grad.detach().numpy()[0]-self.alpha1_grad_sum, self.alpha2_tch.grad.detach().numpy()[0]-self.alpha2_grad_sum, self.alpha3_tch.grad.detach().numpy()[0]-self.alpha3_grad_sum, self.k_tch.grad.detach().numpy()-self.k_grad_sum]#, self.alpha2_tch[0].grad.detach().numpy()[0]-self.alpha2_grad_sum]
            self.alpha1_grad_sum = self.alpha1_grad_sum + grad4[0]
            self.alpha2_grad_sum = self.alpha2_grad_sum + grad4[1]
            self.alpha3_grad_sum = self.alpha3_grad_sum + grad4[2]
            self.k_grad_sum = self.k_grad_sum + grad4[3]

            self.index_tensors.append(loss1.detach().numpy()[0])
            self.index_tensors.append(loss2.detach().numpy()[0])            
            self.index_tensors.append(loss3.detach().numpy()[0])
            self.index_tensors.append(loss4.detach().numpy()[0])
            
            self.index_tensor_grads.append(grad1)
            self.index_tensor_grads.append(grad2)
            self.index_tensor_grads.append(grad3)
            self.index_tensor_grads.append(grad4)

            return False, const_index

        # exit()
        # print("solution",solution)
        solution.retain_grad()
        return solution[0], False  # A tensor

    # def mdp_layer():
    #     '''

    #     '''


    def updateParameters(self,rewards):

        # finr direction for improving performance
        objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
        for index, value in enumerate(rewards): #index from 0 to horizon -1
            objective_tensor = objective_tensor + value            
        objective_tensor.backward(retain_graph=True)

        # Get Gradients
        alpha1_grad = self.alpha1_tch.grad.detach().numpy() - self.alpha1_grad_sum
        alpha2_grad = self.alpha2_tch.grad.detach().numpy() - self.alpha2_grad_sum
        alpha3_grad = self.alpha3_tch.grad.detach().numpy() - self.alpha3_grad_sum
        k_grad = self.k_tch.grad.detach().numpy() - self.k_grad_sum

        self.alpha1_grad_sum = self.alpha1_grad_sum + alpha1_grad
        self.alpha2_grad_sum = self.alpha2_grad_sum + alpha2_grad
        self.alpha3_grad_sum = self.alpha3_grad_sum + alpha3_grad
        self.k_grad_sum = self.k_grad_sum + k_grad

        print(objective_tensor, alpha1_grad, alpha2_grad, alpha3_grad, k_grad)

        ## TODO: write code for constrained GD here
        self.alpha1_nominal = self.alpha1_nominal + self.beta*alpha1_grad
        self.alpha2_nominal = self.alpha2_nominal + self.beta*alpha2_grad
        self.alpha3_nominal = self.alpha3_nominal + self.beta*alpha3_grad
        self.k_nominal = self.k_nominal + self.beta*k_grad

        # clipping > 0
        if self.alpha1_nominal < 0:
            self.alpha1_nominal = 0
        if self.alpha2_nominal < 0:
            self.alpha2_nominal = 0
        if self.alpha3_nominal < 0:
            self.alpha3_nominal = 0
        if self.k_nominal < 0:
            self.k_nominal = 0
        print(f"alpha1_nom:{self.alpha_nominal}, alpha2_nom:{self.alpha2_nominal}, alpha3_nom:{self.alpha3_nominal} k_nominal:{self.k_nominal}")

        episode_reward = objective_tensor.detach().numpy()
        return episode_reward

    def updateParameterFeasibility(self,rewards,indexes):
        # No need to make new tensors here

        # find feasible direction with QP first
        N = len(self.index_tensors) - 2
        d = cp.Variable((2,1))
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

    #Set policy agent
    print("HERE")
    actor = Actor(alpha=args.alpha, alpha1 = args.alpha1, alpha2=args.alpha2, alpha3=args.alpha3, k=args.k, horizon=args.horizon, beta=args.lr_actor)
    dynam = torch_dynamics.apply

    episode_rewards = []
    parameters = []
    parameters.append([args.alpha1, args.alpha2])
    ftime = []

    for ep in range(args.total_episodes):

        # Initialize tensor list
        agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
        agentT = SingleIntegrator(np.array([1,0]),dt,ax,0)
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

            # s_time = time.time()
            u, indexes = actor.policy(x, agentF,agentT,horizon_step)  # tensor    
            if False in u:
                print("QP Failed at ", horizon_step)
                ftime.append(horizon_step)
                break
            # print("time policy",time.time()-s_time)
            # s_time = time.time()
            u.retain_grad()
            x_ = dynam(x,u)  
            x_.retain_grad()
            # print("time dynamics",time.time()-s_time)
            # print(f"x_:{x_}, u:{u}")   
            
            # Store state and input tensors
            state_tensors.append(x_)
            input_tensors.append(u)
            rewards.append( agentF.compute_reward_tensor(x_,agentT) )

            # usum = u.sum()
            # usum.backward(retain_graph=True)
            
            # xsum = x_.sum()
            # xsum.backward(retain_graph=True)            
            
            # alpha1_grad = actor.alpha1_tch[0].grad.detach().numpy()[0] - actor.alpha1_grad_sum
            # alpha2_grad = actor.alpha2_tch[0].grad.detach().numpy()[0] - actor.alpha2_grad_sum
            # alpha3_grad = actor.alpha3_tch[0].grad.detach().numpy()[0] - actor.alpha3_grad_sum
        
            # alpha1_grad_1 = actor.alpha1_tch[-1].grad.detach().numpy()[0]
            # alpha2_grad_1 = actor.alpha2_tch[-1].grad.detach().numpy()[0]
            # alpha3_grad_1 = actor.alpha3_tch[-1].grad.detach().numpy()[0]              
            
            # actor.alpha1_grad_sum += alpha1_grad; actor.alpha2_grad_sum += alpha2_grad; actor.alpha3_grad_sum += alpha3_grad;
            # print("alpha1 grad solution", alpha1_grad, alpha2_grad, alpha3_grad)
            # print("alpha1_1 grad solution", alpha1_grad_1, alpha2_grad_1, alpha3_grad_1)
            # print("u grad",u.grad)
            # print("x grad solution", x.grad)


            # reward = rewards[-1].sum()
            # reward.backward(retain_graph=True)
            # alpha1_grad = actor.alpha1_tch[0].grad.detach().numpy()[0] - actor.alpha1_grad_sum
            # actor.alpha1_grad_sum += alpha1_grad
             
            # print("alpha1 grad reward", alpha1_grad)
            # print("x grad reward", x_.grad)
            # exit()

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
        # print("done")

        if indexes == False:
            print("Parameter Successful! -> Improving PERFORMANCE")
            success = actor.updateParameters(rewards)
            if not success:
                break
            parameters.append([actor.alpha1_nominal, actor.alpha2_nominal])
            actor.resetParams()
        else:
            # Update parameter feasibility
            print("Parameter Unsuccessful -> Improving FEASIBILITY")
            success = actor.updateParameterFeasibility(rewards, indexes)
            if not success:
                break
            parameters.append([actor.alpha1_nominal, actor.alpha2_nominal])
            actor.resetParams()
       
        # sum_reward = actor.updateParameters(rewards)
        # episode_rewards.append(sum_reward)
    
        # actor.updateParameters(rewards)
    



parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="UnicycleFollower")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--lr_actor', type=float, default=0.03, metavar='G',help='learning rate of actor')  #0.003
parser.add_argument('--lr-critic', type=float, default=0.03, metavar='G',help='learning rate of critic') #0.003
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='batch size (default: 256)') #100
parser.add_argument('--buffer-capacity', type=int, default=20, metavar='N', help='buffer_capacity') #10
parser.add_argument('--max-steps', type=int, default=200, metavar='N',help='maximum number of steps of each episode') #70
parser.add_argument('--total-episodes', type=int, default=10, metavar='N',help='total training episodes') #1000
parser.add_argument('--policy-freq', type=int, default=500, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
parser.add_argument('--horizon', type=int, default=50, metavar='N',help='RL time horizon') #3
parser.add_argument('--alpha', type=float, default=0.15, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha1', type=float, default=1.0, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha2', type=float, default=0.7, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--alpha3', type=float, default=0.7, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--train', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie_name', default="test.mp4")
parser.add_argument('--dt', type=float, default="0.01")
args = parser.parse_args("")


train(args)
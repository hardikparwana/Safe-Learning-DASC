import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from dynamics import *

dynam = torch_dynamics.apply

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

def getGrad(param):
    if param.grad==None:
        return 0
    try:
        value = param.grad.detach().numpy()[0]
    except:
        value = param.grad.detach().numpy()
    param.grad = None
    return value  

def getParamGrads(mpc):
    alpha1_grad = getGrad(mpc.one_step_forward.alpha1_tch)
    alpha2_grad = getGrad(mpc.one_step_forward.alpha2_tch)
    alpha3_grad = getGrad(mpc.one_step_forward.alpha3_tch)
    k_grad = getGrad(mpc.one_step_forward.k_tch)
    grads = np.array([[alpha1_grad, alpha2_grad, alpha3_grad, k_grad]])          
    return grads    

def updateParameterPerformanceUnconstrained(mpc,params,beta,rewards,cons):
    
    # find direction for improving performance
    objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
    for index, value in enumerate(rewards): #index from 0 to horizon -1
        objective_tensor = objective_tensor + value            
    objective_tensor.backward(retain_graph=True)
    
    #Get gradients    
    grads_objective = getParamGrads(mpc)  
    
    params = params + beta*grads_objective        
    return params

def updateParameterPerformance(mpc,params,beta,rewards,cons):
    
    #### Find direction for improving performance
    objective_tensor = torch.tensor(0,requires_grad=True, dtype=torch.float)
    for index, value in enumerate(rewards): #index from 0 to horizon -1
        objective_tensor = objective_tensor + value            
    objective_tensor.backward(retain_graph=True)
    
    print(f"Horizon Reward: {objective_tensor.detach().numpy()}")
    
    #Get gradients    
    grads_objective = getParamGrads(mpc)    
    
    n_param = np.shape(params)[0]
    n_cons = len(cons)

    d = cp.Variable((n_param,1))
    const = []
    objective = cp.Maximize( grads_objective @ d  )

    ### Find constrained directions       
    for index, value in enumerate(cons):
             value.sum().backward(retain_graph=True)
             curr_cons = value.detach().numpy()[0] if value.detach().numpy()[0]>0.0 else 0.0
             grads = getParamGrads(mpc)
             const += [ curr_cons + grads @ d >= 0 ]
    const += [cp.norm(d,"inf")<=0.3]

    # Solve the QP         
    problem = cp.Problem(objective, const)
    problem.solve()
    if problem.status in ["infeasible", "unbounded"]: #problem.status!='optimal':
        print("No Feasible Direction found!")
        return False, False

    # Update Params         
    params = params + beta*d.value.reshape(1,-1)[0]        
    return True, params
    
def updateParameterFeasibility(mpc,params,beta,feasible_cons,infeasible_cons):
    
    n_param = np.shape(params)[0]
    n_feasible_cons = len(feasible_cons)
    n_infeasible_cons = len(infeasible_cons)
    
    d = cp.Variable((n_param,1))
    const = []
    
    ### Objective: 
    grads_objective = np.zeros((1,n_param))
    for index, value in enumerate(infeasible_cons):
        value.sum().backward(retain_graph=True)
        grads_objective = grads_objective + getParamGrads(mpc)    
    objective = cp.Maximize( grads_objective @ d  )
    
    ### Find constrained directions       
    for index, value in enumerate(feasible_cons):
             value.sum().backward(retain_graph=True)
             curr_cons = value.detach().numpy()[0] if value.detach().numpy()[0]>0.0 else 0.0
             grads = getParamGrads(mpc)
             const += [ curr_cons + grads @ d >= 0 ]
    const += [cp.norm(d,"inf")<=0.3]
    
    # Solve the QP         
    problem = cp.Problem(objective, const)
    problem.solve()
    if problem.status!='optimal':
        print("No Feasible Direction found!")
        return False, False
    
    # Update Params         
    params = params + beta*d.value.reshape(1,-1)[0]     
    return True, params


class closedLoopLayer(nn.Module):

    def __init__(self,agent_class,target_class,alpha1,alpha2,alpha3,k):  # pass in unicycle and target here instead of taking it from globla header declarations
        super().__init__()

        self.agent_nominal_controller = agent_class.nominal_controller_tensor2
        self.target_xdot = target_class.xdotTensor

        self.CLF_loss_tensor = agent_class.CLF_loss_tensor_simple
        self.CBF1_loss_tensor = agent_class.CBF1_loss_tensor_simple
        self.CBF2_loss_tensor = agent_class.CBF2_loss_tensor_simple
        self.CBF3_loss_tensor = agent_class.CBF3_loss_tensor_simple

        self.alpha1_tch = torch.tensor(alpha1, requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor(alpha2, requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor(alpha3, requires_grad=True, dtype=torch.float)
        self.k_tch = torch.tensor(k, requires_grad=True, dtype=torch.float)

        self.solver_args = {
            'verbose': False#,
            #'max_iters': 1000000
        }
        
    def updateParams(self,params):
        self.alpha1_tch = torch.tensor(params[0], requires_grad=True, dtype=torch.float)
        self.alpha2_tch = torch.tensor(params[1], requires_grad=True, dtype=torch.float)
        self.alpha3_tch = torch.tensor(params[2], requires_grad=True, dtype=torch.float)
        self.k_tch = torch.tensor(params[3], requires_grad=True, dtype=torch.float)

    def forward(self,x_agent,x_target,u_target):
        
        U_d_ = self.agent_nominal_controller(x_agent,x_target)
        target_xdot_ = self.target_xdot(x_target,u_target)  ######################### target.U??

        # simplified define tensors for cvx DPP form
        V_,dV_dxA_f_, dV_dxA_g_,dV_dxB_temp = self.CLF_loss_tensor(x_agent,x_target)
        h1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_temp = self.CBF1_loss_tensor(x_agent,x_target)
        h2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_temp = self.CBF2_loss_tensor(x_agent,x_target)
        h3_,dh3_dxA_f_, dh3_dxA_g_,dh3_dxB_temp = self.CBF3_loss_tensor(x_agent,x_target)

        # print(f"dV_dxB_temp:{dV_dxB_temp}, target_xdot_:{target_xdot_}, dV type: {target_xdot_.dtype}")
        
        dV_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh1_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh2_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh3_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_

        ah1_ = self.alpha1_tch*h1_  
        ah2_ = self.alpha2_tch*h2_  
        ah3_ = self.alpha3_tch*h3_  
        kV_ = self.k_tch * V_  

        try: 
            # Find the controller
            solution, = cvxpylayer(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=self.solver_args)

            # Tensors for constraint derivative
            cbf1 = dh1_dxA_f_ + torch.matmul(dh1_dxA_g_, solution) + dh1_dxB_target_xdot_ + ah1_
            cbf2 = dh2_dxA_f_ + dh2_dxA_g_ @ solution + dh2_dxB_target_xdot_ + ah2_ 
            cbf3 = dh3_dxA_f_ + dh3_dxA_g_ @ solution + dh3_dxB_target_xdot_ + ah3_
            u11 = solution[0,0]+u_max[0]; u12 = u_max[0]-solution[0,0]
            u21 = solution[0,1]+u_max[1]; u22 = u_max[1]-solution[0,1]
            # lyap = -dV_dxA_f_ - dV_dxA_g_ @ solution - dV_dxB_target_xdot_ - kV_ 

            return solution[0], [cbf1, cbf2, cbf3, u11, u12, u21, u22]
        except:
            # Find required tensors and return them
            # solution1, = cvxpylayer_relaxed1(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=self.solver_args)
            # solution2, = cvxpylayer_relaxed2(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=self.solver_args)
            # solution3, = cvxpylayer_relaxed3(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=self.solver_args)

            # Tensors for constraint derivative
            cbf1 = dh1_dxA_f_ + dh1_dxB_target_xdot_ + ah1_
            cbf2 = dh2_dxA_f_ + dh2_dxB_target_xdot_ + ah2_ 
            cbf3 = dh3_dxA_f_ + dh3_dxB_target_xdot_ + ah3_
            
            return [cbf1, cbf2, cbf3]


class horizonControl(nn.Module):

    def __init__(self,params,horizon):
        super().__init__()
        self.one_step_forward = closedLoopLayer(Unicycle2D,SingleIntegrator,params[0],params[1],params[2],params[3])  # pass agent and target functions here
        self.horizon = horizon
        self.reward = Unicycle2D.compute_reward_tensor_simple

    def forward(self,x_agent,x_target,u_target):
        '''
            x_agent: initial location of agent
            x_target: array of future target locations
        '''

        states = [x_agent]
        inputs = []
        rewards = []
        constraints = []
        infeasible_constraints = []

        try: 
            for i in range(self.horizon):
                u, cons = self.one_step_forward(states[i],x_target[i],u_target[i].reshape(-1,1) )
                x_1 = dynam(states[i],u)
                r = self.reward(x_1,x_target[i+1])
                if r.detach().numpy()<0:  # issue with discrete time!!!!
                    infeasible_constraints = infeasible_constraints + [r]
                    break
                states.append(x_1); inputs.append(u); rewards.append(r); constraints = constraints + cons
        except:
            # solve the last one but get required tensors
            cons = self.one_step_forward(states[i],x_target[i],u_target[i].reshape(-1,1) )
            infeasible_constraints = infeasible_constraints + cons
        
        return states, \
            inputs, \
            rewards, \
            constraints, \
            infeasible_constraints 
                



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
    dt = args.dt
    
    params = np.array([args.alpha1, args.alpha2, args.alpha3, args.k])
    mpc = horizonControl(params,args.horizon)
    
    # Initialize agents
    agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
    agentT = SingleIntegrator(np.array([1,0]),dt,ax,0)
    
    forward = False
    current_horizon = 0
    episode_time = 0

    for ep in range(args.episodes):
        
        print(f"params:{params}")
        
        episode_time = episode_time + 1 if forward == True else episode_time      
         
        # Initialize tensor list               
        Fx = torch.tensor(agentF.X.reshape(-1,1),dtype=torch.float, requires_grad=True)
        TXs = [torch.tensor(agentT.X, dtype=torch.float)]
        TUs = []

        # Produce leader trajectory only
        for horizon_step in range(args.horizon):
            uL = 0.5
            vL = 2.6*np.sin(np.pi*(episode_time+horizon_step)*dt) #  0.1 # 1.2
            
            TUs.append(torch.tensor(np.array([uL,vL]),dtype=torch.float))
            agentT.step(uL,vL) 
            TXs.append(torch.tensor(agentT.X, dtype=torch.float))
        agentT.X = TXs[0].detach().numpy() 
        ## Forward it to Network        
        x, u, r, cons, infeasible_cons = mpc(Fx,TXs,TUs)
        success = len(infeasible_cons)==0           

        # Update Parameters
        if success:
            print("Parameter Successful! -> Improving PERFORMANCE")
            print(f"current horizon:{len(x)-1}")
            update, params = updateParameterPerformance(mpc,params,args.lr_beta,r,cons)  
            if update:
                mpc.one_step_forward.updateParams(params)
                forward = True
                current_horizon = args.horizon
            else:
                exit()

        else: 
            print("Parameter Unsuccessful -> Improving FEASIBILITY")
            print(f"current horizon:{len(x)-1}")
            if (len(x)-1)>current_horizon or (len(x)-1)==args.horizon:
                current_horizon = len(x)-1
                forward = True
            else:
                forward = False
            update, params = updateParameterFeasibility(mpc,params,args.lr_beta,cons,infeasible_cons)  
            if update:
                mpc.one_step_forward.updateParams(params)
            else:
                exit()
        
        if len(u)>0:
            # print(f"u:{u[0].detach().numpy()}")
            # Move agent one step        
            agentF.step(u[0].detach().numpy())  
            agentT.step(TUs[0][0],TUs[0][1]) 

            # Visulaize one step motion with animation plot
            if args.animate_episode:
                lines, areas, bodyF = agentF.render(lines,areas,bodyF)
                bodyT = agentT.render(bodyT)
                bodyF_tensor.set_offsets([x[1].detach().numpy()[0,0],x[1].detach().numpy()[1,0]])        
                fig.canvas.draw()
                fig.canvas.flush_events()     
        
            # Visualize predicted horizon
            if args.animate_horizon:
                for i in range(len(x)):
                    lines, areas, bodyF = agentF.render_state(lines,areas,bodyF,x[i].detach().numpy())
                    bodyT = agentT.render_state(bodyT,TXs[i].detach().numpy())
                    bodyF_tensor.set_offsets([x[i].detach().numpy()[0,0],x[i].detach().numpy()[1,0]])        
                    fig.canvas.draw()
                    fig.canvas.flush_events()    
        print(f"forward: {forward}, start time:{episode_time}")
        # print("DONE")


parser = argparse.ArgumentParser(description='adaptive_qp')
parser.add_argument('--alpha1', type=float, default=1.0, metavar='G',help='CBF parameter')  
parser.add_argument('--alpha2', type=float, default=0.7, metavar='G',help='CBF parameter')  
parser.add_argument('--alpha3', type=float, default=0.7, metavar='G',help='CBF parameter')  
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter') 
parser.add_argument('--episodes', type=int, default=200, metavar='N',help='total training episodes') 
parser.add_argument('--horizon', type=int, default=10, metavar='N',help='total training episodes')
parser.add_argument('--lr_beta', type=float, default=1.0, metavar='G',help='learning rate of parameter')  #0.003
parser.add_argument('--dt', type=float, default="0.01")
parser.add_argument('--animate_episode', type=bool, default=True)
parser.add_argument('--animate_horizon', type=bool, default=False)

args = parser.parse_args("")
train(args)
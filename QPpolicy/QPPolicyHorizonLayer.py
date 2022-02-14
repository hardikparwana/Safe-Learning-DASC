import torch
import torch.nn as nn
import numpy as np
from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from dynamics import *
import time

dynam = torch_dynamics.apply

def getGrad(param):
    value = param.grad.detach().numpy()[0]
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
    
    #Get gradients    
    grads_objective = getParamGrads(mpc)    
    
    n_param = np.shape(params)[0]
    n_cons = len(cons)
    
    d = cp.Variable((n_param,1))
    const = []
    objective = cp.Minimize( grads_objective @ d  )
    
    ### Find constrained directions       
    for index, value in enumerate(cons):
             value.sum().backward()
             grads = getParamGrads(mpc)
             const += [ value + grads @ d >= 0 ]
    
    # Solve the QP         
    problem = cp.Problem(objective, const)
    problem.solve()
    if problem.status!='optimal':
        print("No Feasible Direction found!")
        return False, 
    
    # Update Params         
    params = params + beta*grads    
    
    return params
    
def updateParameterFeasibility(mpc,params,beta,feasible_cons,infeasible_cons):
    
    n_param = np.shape(params)[0]
    n_feasible_cons = len(feasible_cons)
    n_infeasible_cons = len(infeasible_cons)
    
    d = cp.Variable((n_param,1))
    const = []
    
    ### Objective: 
    grads_objective = np.zeros((1,n_param))
    for index, value in enumerate(infeasible_cons):
        value.sum().backward()
        grads_objective = grads_objective + getParamGrads(mpc)    
    objective = cp.Minimize( grads_objective @ d  )
    
    ### Find constrained directions       
    for index, value in enumerate(cons):
             value.sum().backward()
             grads = getParamGrads(mpc)
             const += [ value + grads @ d >= 0 ]
    
    # Solve the QP         
    problem = cp.Problem(objective, const)
    problem.solve()
    if problem.status!='optimal':
        print("No Feasible Direction found!")
        return False, 
    
    # Update Params         
    params = params + beta*grads    
    
    return params


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
            'verbose': False,
            'max_iters': 1000000
        }

    def forward(self,x_agent,x_target,u_target):

        U_d_ = self.agent_nominal_controller(x_agent,x_target)
        target_xdot_ = self.target_xdot(x_target,u_target)  ######################### target.U??

        # simplified define tensors for cvx DPP form
        V_,dV_dxA_f_, dV_dxA_g_,dV_dxB_temp = self.CLF_loss_tensor(x_agent,x_target)
        h1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_temp = self.CBF1_loss_tensor(x_agent,x_target)
        h2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_temp = self.CBF2_loss_tensor(x_agent,x_target)
        h3_,dh3_dxA_f_, dh3_dxA_g_,dh3_dxB_temp = self.CBF3_loss_tensor(x_agent,x_target)

        dV_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh1_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh2_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
        dh3_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_

        ah1_ = self.alpha1_tch*h1_  
        ah2_ = self.alpha2_tch*h2_  
        ah3_ = self.alpha3_tch*h3_  
        kV_ = self.k_tch * V_  

        # Find the controller
        solution, = cvxpylayer(U_d_, ah1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_target_xdot_, ah2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_target_xdot_, ah3_,dh3_dxA_f_, dh3_dxA_g_, dh3_dxB_target_xdot_,  kV_,dV_dxA_f_,dV_dxA_g_ ,dV_dxB_target_xdot_, solver_args=self.solver_args)

        # Tensors for constraint derivative
        cbf1 = dh1_dxA_f_ + torch.matmul(dh1_dxA_g_, solution) + dh1_dxB_target_xdot_ + ah1_
        cbf2 = dh2_dxA_f_ + dh2_dxA_g_ @ solution + dh2_dxB_target_xdot_ + ah2_ 
        cbf3 = dh3_dxA_f_ + dh3_dxA_g_ @ solution + dh3_dxB_target_xdot_ + ah3_
        # lyap = -dV_dxA_f_ - dV_dxA_g_ @ solution - dV_dxB_target_xdot_ - kV_ 

        return solution[0], [cbf1, cbf2, cbf3]


class horizonControl(nn.Module):

    def __init__(self,):
        super().__init__(params,horizon)
        self.one_step_forward = closedLoopLayer(Unicycle2D,SingleIntegrator,params[0],params[1],params[2],params[3])  # pass agent and target functions here
        self.horizon = horizon


    def forward(self,x_agent,x_target,u_target):
        '''
            x_agent: initial location of agent
            x_target: array of future target locations
        '''

        states = [x_agent]
        inputs = []
        rewards = []
        constraints = []
        
        for i in range(self.horizon):
            u, cons = self.one_step_forward(states[i],x_target[i],u_target[i].reshape(-1,1),dtype=torch.float ))
            x_1 = dynam(states[i],u)
            r = reward(x_1,x_target[i+1])
            states.append(x_1); inputs.append(u); rewards.append(r); constraints = constraints + cons
            
        return states, \
            inputs, \
            constraints, \
            rewards 

        # Time 1:
        # s_time = time.time()
        u0, cons0 = self.one_step_forward(x_agent,x_target[0],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
        x1 = dynam(x_agent,u0)
        r1 = reward(x1,x_target[1])

        #Time 2: 
        u1, cons1 = self.one_step_forward(x1,x_target[1],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
        x2 = dynam(x1,u1)

        #Time 3: 
        u2, cons2 = self.one_step_forward(x2,x_target[2],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
        x3 = dynam(x2,u2)

        #Time 4: 
        u3, cons3 = self.one_step_forward(x1,x_target[1],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
        x4 = dynam(x1,u3)

        return True, [x1, x2, x3, x4], \
                [u0, u1, u2, u3], \
                [cons0, cons1, cons2, cons3]
                



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
    
    params = np.array([args.alpha1, args.alpha2, args.alpha3, args.k])
    
    # Initialize tensor list
    agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
    agentT = SingleIntegrator(np.array([1,0]),dt,ax,0)

    for ep in range(args.episodes):
       
        Fx = torch.tensor(agentF.X.reshape(-1,1),dtype=torch.float, requires_grad=True)
        Tx = torch.tensor(agentT.X.reshape(-1,1),dtype=torch.float, requires_grad=True)
        
        # produce leader trajectory only
        TXs = [agentT.X]
        TUs = []
        for horizon_step in range(args.horizon):
            uL = 0.5
            vL = 2.6*np.sin(np.pi*t_roll) #  0.1 # 1.2
            
            TUs.append(np.array([uL,vL]))
            agentT.step(uL,vL) 
            TXs.append(agentT.X)
            
        ## Forward it to Network        
        mpc = horizonControl()
        success, x, u, cons = mpc(Fx,[Tx,Tx,Tx,Tx])
        
        
        
        if success:
            print("Parameter Successful! -> Improving PERFORMANCE")
            updateParameterPerformance(mpc,params,args.lr_beta)


parser = argparse.ArgumentParser(description='adaptive_qp')
args = parser.parse_args("")
parser.add_argument('--alpha1', type=float, default=1.0, metavar='G',help='CBF parameter')  
parser.add_argument('--alpha2', type=float, default=0.7, metavar='G',help='CBF parameter')  
parser.add_argument('--alpha3', type=float, default=0.7, metavar='G',help='CBF parameter')  
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter') 
parser.add_argument('--episodes', type=int, default=1, metavar='N',help='total training episodes') 
parser.add_argument('--horizon', type=int, default=100, metavar='N',help='total training episodes') v
parser.add_argument('--lr_beta', type=float, default=0.03, metavar='G',help='learning rate of parameter')  #0.003
parser.add_argument('--dt', type=float, default="0.01")

train(args)
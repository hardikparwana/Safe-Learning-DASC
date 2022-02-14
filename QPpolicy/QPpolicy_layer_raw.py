import torch
import torch.nn as nn
import numpy as np
from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
from dynamics import *
import time

# class qp_policy(torch.autograd.Function):

# 	@staticmethod
# 	def forward(ctx, input_x, input_target):
# 		'''
# 			input_x: (n_agent x 1) agent initial state
# 			input_leader: (n_leader x T) leader trajectpry over next T time steps
# 		'''

# 		U_d_ = agent.nominal_controller_tensor(input_x,input_target)
# 		target_xdot_ = SingleIntegrator.xdot(target.U)

# 		# simplified define tensors for cvx DPP form
#         V_,dV_dxA_f_, dV_dxA_g_,dV_dxB_temp = Unicycle2D.CLF_loss_tensor(X,target)
#         h1_,dh1_dxA_f_, dh1_dxA_g_,dh1_dxB_temp = Unicycle2D.CBF1_loss_tensor(X,target)
#         h2_,dh2_dxA_f_, dh2_dxA_g_,dh2_dxB_temp = Unicycle2D.CBF2_loss_tensor(X,target)
#         h3_,dh3_dxA_f_, dh3_dxA_g_,dh3_dxB_temp = Unicycle2D.CBF3_loss_tensor(X,target)

#         dV_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
#         dh1_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
#         dh2_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_
#         dh3_dxB_target_xdot_ = dV_dxB_temp @ target_xdot_

		
# 		#save tensors for gradient computation
# 		ctx.save_for_backward(  )

# 		return solution[0]


# 	@staticmethod
# 	def backward(ctx, grad_output):


dynam = torch_dynamics.apply


class closedLoopLayer(nn.Module):

    def __init__(self,agent_class,target_class,alpha1,alpha2,alpha3,k):  # pass in unicycle and target here instead of taking it from globla header declarations
        super().__init__()

        self.agent_nominal_controller = agent_class.nominal_controller_tensor2
        self.target_xdot = target_class.xdotTensor

        self.CLF_loss_tensor = agent_class.CLF_loss_tensor_simple
        self.CBF1_loss_tensor = agent_class.CBF1_loss_tensor_simple
        self.CBF2_loss_tensor = agent_class.CBF2_loss_tensor_simple
        self.CBF3_loss_tensor = agent_class.CBF3_loss_tensor_simple

        self.alpha1_tch = alpha1
        self.alpha2_tch = alpha2
        self.alpha3_tch = alpha3
        self.k_tch = k

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
		super().__init__()
		self.one_step_forward = closedLoopLayer(Unicycle2D,SingleIntegrator,1.5,1.0,1.0,0.2)  # pass agent and target functions here


	def forward(self,x_agent,x_target):
		'''
			x_agent: initial location of agent
			x_target: array of future target locations
		'''

		# Time 1:
		# s_time = time.time()
		u0, cons0 = self.one_step_forward(x_agent,x_target[0],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
		# print("Time: ", time.time()-s_time)
		
		# s_time = time.time()
		x1 = dynam(x_agent,u0)
        r1
		# print("Time: ", time.time()-s_time)

		#Time 2: 
		u1, cons1 = self.one_step_forward(x1,x_target[1],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
		x2 = dynam(x1,u1)

		#Time 3: 
		u2, cons2 = self.one_step_forward(x2,x_target[2],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
		x3 = dynam(x2,u2)

		#Time 4: 
		u3, cons3 = self.one_step_forward(x1,x_target[1],torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ))
		x4 = dynam(x1,u3)

		return [x1, x2, x3, x4], \
			   [u0, u1, u2, u3], \
			   [cons0, cons1, cons2, cons3]


Fx = torch.tensor(np.array([0.0,0.2,0.0]).reshape(-1,1),dtype=torch.float, requires_grad=True)
Tx = torch.tensor(np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float, requires_grad=True)

# Test Closed Loop

# model = closedLoopLayer(Unicycle2D,SingleIntegrator,1.5,1.0,1.0,0.2)
# model(Fx,Tx,torch.tensor( np.array([1.0,0.0]).reshape(-1,1),dtype=torch.float ) )


mpc = horizonControl()

s_time = time.time()
x, u, cons = mpc(Fx,[Tx,Tx,Tx,Tx])
print("Time: ", time.time()-s_time)


print(x[0])

x[0].sum().backward(retain_graph=True)

print(u)

print(cons)
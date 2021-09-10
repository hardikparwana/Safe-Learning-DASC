import torch
import numpy as np
from robot_models.Unicycle2D import *


class torch_dynamics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_x, input_u):

        # x_{t+1} = f(x_t) + g(x_t)u_t
        x = input_x
        u = input_u
        fx = fx_(x)  
        gx = gx_(x)  
        
        # for gradient computation
        df_dx =  df_dx_(x) 
        dgxu_dx = dgxu_dx_(x)  
            
        # save tensors for use in backward computation later on
        ctx.save_for_backward(x,u, gx, df_dx, dgxu_dx)

        return fx + torch.matmul(gx, u)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        grad_output is column vector. Math requires it to be row vector so need adjustment in returning the values
        '''

        input_x, input_u, gx, df_dx, dgxu_dx, = ctx.saved_tensors
        x = input_x.detach().numpy()
        u = input_u.detach().numpy()

        n_x = np.shape(x)[0]
        n_u = np.shape(u)[0]
               
        print(grad_output)
        
        gradient_x = df_dx + dgxu_dx
        gradient_u =  gx 

        output_grad_x = torch.reshape(torch.matmul(torch.transpose(grad_output,0,1),gradient_x),(n_x,1))
        output_grad_u = torch.reshape(torch.matmul(torch.transpose(grad_output,0,1),gradient_u),(n_u,1))

        return output_grad_x, output_grad_u

dtype = torch.float
device = torch.device("cpu")

dynam = torch_dynamics.apply
x = torch.tensor(np.array([1,1]).reshape(-1,1),dtype=dtype, requires_grad=True)
u = torch.tensor(np.array([3,2]).reshape(-1,1),dtype=dtype, requires_grad=True)

# input = torch.tensor(np.append(x,u,axis=0),dtype=dtype, requires_grad=True)

y_pred = dynam(x,u)
print(y_pred)
loss = y_pred.sum()
loss.backward()
print("x grad", x.grad)
print("u grad", u.grad)
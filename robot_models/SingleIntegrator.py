import numpy as np
import torch

def fx_(x):
    return torch.tensor(np.array([0,0]).reshape(-1,1),dtype=torch.float)

def gx_(x):
    g_ = np.array([ [1,0],[0,1]])
    return torch.tensor(g_,dtype=torch.float)

def df_dx_(x, type = 'tensor'):
    dfdx = np.array([[0,0],[0,0]])
    if type == 'numpy':
        return dfdx
    if type == 'tensor':
        return torch.tensor(dfdx,dtype=torch.float)
    
def dgxu_dx_(x, u, type='tensor'):
    u_ = u.detach().numpy()
    dgxudx = np.array([[0, 0],[0, 0]])
    if type == 'numpy':
        return dgxudx
    if type == 'tensor':
        return torch.tensor(dgxudx,dtype=torch.float)

class SingleIntegrator:
    
    def __init__(self,X0,dt,ax,id):
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.V = np.array([0,0]).reshape(-1,1)
        self.dt = dt
        self.t0 = 0
        self.speed = 0
        self.theta = 0
        self.id = id

        self.f = np.array([0,0]).reshape(-1,1)
        self.g = np.array([ [1, 0],[0, 1] ])
        self.U = np.array([0,0]).reshape(-1,1)

        self.body = ax.scatter([],[],c=('r' if self.id==0 else 'g'),s=10)
        self.render_plot()
        
    def step(self,ux,uy): #Just holonomic X,T acceleration

        self.U = np.array([ux,uy]).reshape(-1,1)
        self.X = self.X + ( self.f + self.g @ self.U )*self.dt
        return self.X

    def step_sim(self,a,alpha): #Just holonomic X,T acceleration
        
        X_sim = self.X + np.array([a,alpha])*self.dt
        return X_sim
    
    def render(self,body):

        x = np.array([self.X[0,0],self.X[1,0]])

        # scatter plot update
        body.set_offsets([x[0],x[1]])

        return body

    def render_plot(self):
    
        x = np.array([self.X[0,0],self.X[1,0]])

        # scatter plot update
        self.body.set_offsets([x[0],x[1]])

    def xdot(self,U):
        return torch.tensor( self.f + self.g @ U, dtype=torch.float )
    
    def xdotTensor(X,U):
        return fx_(X) + gx_(X) @ U

    def agentBarrier(self,target,min_D):
        # min distance
        h = np.linalg.norm( self.X - target.X )**2 - min_D**2
        factor = 2*( self.X - target.X )
        dh_dx_agent = np.array([factor[0,0], factor[1,0]])
        dh_dx_target = -dh_dx_agent
        return h, dh_dx_agent, dh_dx_target

    def obsBarrier(self,target,min_D):
        # min distance
        h = np.linalg.norm( self.X - target.X )**2 - min_D**2
        # print(f"agent:{self.X}, obs:{target.X}, h:{h}")
        dh_dx_agent = np.array([2*( self.X - target.X )[0,0], 2*( self.X - target.X )[1,0]] )
        return h, dh_dx_agent

    def connectivityBarrier(self,target,max_D):
        # max distance
        h = max_D**2 - np.linalg.norm( self.X - target.X )**2
        dh_dx_agent = np.array([-2*( self.X - target.X )[0,0], -2*( self.X - target.X )[1,0] ])
        dh_dx_target = -dh_dx_agent
        return h, dh_dx_agent, dh_dx_target

    def Lyapunov(self,target,dist):
        #desired distance
        V = (np.linalg.norm( self.X - target.X ) - dist)**2
        factor = 2*(np.linalg.norm( self.X - target.X ) - dist)/np.linalg.norm( self.X - target.X ) * ( np.array([ self.X[0,0], self.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
        dV_dx_agent = factor
        dV_dx_target = -factor
        return V, dV_dx_agent, dV_dx_target

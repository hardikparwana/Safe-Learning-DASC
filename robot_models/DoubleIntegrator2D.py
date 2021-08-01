import numpy as np

class DoubleIntegrator2D:
    
    def __init__(self,X0,dt):
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.t0 = 0
        self.speed = 0
        self.theta = 0

        self.f = np.array([self.X[2,0],self.X[3,0],0,0]).reshape(-1,1)
        self.g = np.array([ [0, 0],[0, 0],[1, 0], [0, 1] ])

        # acceleration input
        self.U = np.array([0,0]).reshape(-1,1)
        
    def step(self,ax,ay): #Just holonomic X,T acceleration

        self.U = np.array([ax,ay]).reshape(-1,1)
        self.X = self.X + ( self.f + self.g @ self.U )*self.dt
        self.f = np.array([self.X[2,0],self.X[3,0],0,0]).reshape(-1,1)
        return self.X

    def step_sim(self,a,alpha): #Just holonomic X,T acceleration
        
        X_sim = self.X + np.array([a,alpha])*self.dt
        return X_sim
    
    def render(self,body):

        x = np.array([self.X[0,0],self.X[1,0]])

        # scatter plot update
        body.set_offsets([x[0],x[1]])

        return body

    def xdot(self,U):
        return ( self.f + self.g @ U )
import numpy as np
import torch

def fx_(x):
    return x

def gx_(x):
    return torch.tensor(np.array([[1,0],[0,1]]),dtype=torch.float);

def df_dx_(x, type = 'tensor'):
    if type == 'numpy':
        return np.array(np.array([[1,0],[0,1]]))
    if type == 'tensor':
        return torch.tensor([[1,0],[0,1]],dtype=torch.float)
    
def dgxu_dx_(x, type='tensor'):
    if type == 'numpy':
        return np.array([[0,0],[0,0]])
    if type == 'tensor':
        return torch.tensor([[0,0],[0,0]],dtype=torch.float)

class Unicycle2D:
    
    def __init__(self,X0,dt,fov_length,fov_angle,max_D,min_D):
        X0 = X0.reshape(-1,1)
        self.X  = X0
        self.dt = dt
        self.d1 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16) 
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        self.f = np.array([0,0,0]).reshape(-1,1)
        self.g = np.array([ [np.cos(X0[2,0]),0],[np.sin(X0[2,0]),0],[0,1] ])
        self.f_corrected = np.array([0,0,0]).reshape(-1,1)
        self.g_corrected = np.array([ [0,0],[0,0],[0,0] ])

        self.FoV_length = fov_length
        self.FoV_angle = fov_angle

        self.max_D = max_D
        self.min_D = min_D

        self.h_index = 1
        
    def step(self,U):
        U = U.reshape(-1,1)
        # print("input",U)
        self.g = np.array([ [np.cos(self.X[2,0]),0],[np.sin(self.X[2,0]),0],[0,1] ])

        # disturbance term
        self.d1 = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        
        extra_f = np.array([self.d1*np.cos(self.X[2][0]),self.d2*np.sin(self.X[2][0]),self.d3]).reshape(-1,1)
        extra_g = np.array([ [0,0],[0,0],[0,0] ])
        
        self.X = self.X + (extra_f + extra_g @ U + self.f + self.g @ U)*self.dt
        # print("gU",self.g @ U)
    
        self.X[2,0] = self.wrap_angle(self.X[2,0])
        
        return self.X

    def step_sim(self,U):
        
        U = U.reshape(-1,1)
        g_sim = np.array([ [np.cos(self.X[2,0]),0],[np.sin(self.X[2,0]),0],[0,1] ])

        # disturbance term
        self.d1_sim = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        self.d2_sim = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3_sim = 0.3
        
        extra_f_sim = np.array([self.d1*np.cos(self.X[2][0]),self.d2*np.sin(self.X[2][0]),self.d3]).reshape(-1,1)
        extra_g_sim = np.array([ [0,0],[0,0],[0,0] ])
       
        X_sim = self.X + (extra_f + extra_g @ U + self.f + self.g @ U)*self.dt
    
        X_sim[2] = self.wrap_angle(X_sim[2])
        
        return X_sim
    
    def render(self,lines,areas,body):
        # length = 3
        # FoV = np.pi/3   # 60 degrees

        x = np.array([self.X[0,0],self.X[1,0]])
  
        theta = self.X[2][0]
        theta1 = theta + self.FoV_angle/2
        theta2 = theta - self.FoV_angle/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + self.FoV_length*e1
        P2 = x + self.FoV_length*e2  

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        
        triangle_v = [ x,P1,P2,x ]  

        # lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)

        # scatter plot update
        body.set_offsets([x[0],x[1]])

        return lines, areas, body

    def wrap_angle(self,angle):
        if angle>np.pi:
            angle = angle - 2*np.pi
        if angle<-np.pi:
            angle = angle + 2*np.pi
        return angle

    def xdot(self,U):
        # print("g",self.g + self.g_corrected)
        return ( self.f + self.f_corrected + (self.g + self.g_corrected) @ U )
    
    def CBF_loss(self,target,u,w):

        alpha = 0.1 #1.0 does not work

        h1 , h1_dxA ,_h1_dxB = CBF1_loss(self,target)
        h2 , h2_dxA , h2_dxB = CBF2_loss(self,target)
        h3 , h3_dxA , h3_dxB = CBF3_loss(self,target)

        U = np.array([u,w]).reshape(-1,1)
        h1_ = h1A @ U + h1 + alpha * h1
        h2_ = h2 @ U + h2 + alpha * h2
        h3_ = h3 @ U + h3 + alpha * h3

        if h1_ > 0 and h2_ > 0 and h3_ > 0:
            return True, h1_, h2_, h3_
        else:
            return False, h1_, h2_, h3_

    def CBF1_loss(self,target):
        h = self.max_D**2 - np.linalg.norm( self.X[0:2,0] - target.X[:,0] )**2
        dh_dx_agent = np.append(- 2*( self.X[0:2,0] - target.X[:,0] ),0)
        dh_dx_target = 2*( self.X[0:2,0] - target.X[:,0] )

        # h_dot_B = 2*( self.X[0:2,0] - target.X[:,0] ) @ target.V  - 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.f[0:2,:]  ) - 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.f_corrected[0:2,:]  )
        # h_dot_A = - 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.g[0:2,:]  ) - 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.g_corrected[0:2,:]  )
        
        return h, dh_dx_agent, dh_dx_target  #h_dot_A, h_dot_B

    def CBF1_sim_loss(self,Fx,Tx):
        h = self.max_D**2 - np.linalg.norm( Fx[0:2,0] - Tx[:,0] )**2
        dh_dx_agent = np.append(- 2*( Fx[0:2,0] - Tx[:,0] ),0)
        return dh_dx

    def CBF2_loss(self,target):
        h = np.linalg.norm( self.X[0:2,0] - target.X[:,0] )**2 - self.min_D**2
        dh_dx_agent = np.append(2*( self.X[0:2,0] - target.X[:,0] ),0)
        dh_dx_target = - 2*( self.X[0:2,0] - target.X[:,0] )

        # h_dot_B = - 2*( self.X[0:2,0] - target.X[:,0] ) @ target.V  + 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.f[0:2,:]  ) + 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.f_corrected[0:2,:]  )
        # h_dot_A =   2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.g[0:2,:]  ) + 2*( self.X[0:2,0] - target.X[:,0] ) @ ( self.g_corrected[0:2,:]  )
        return h, dh_dx_agent, dh_dx_target #h_dot_A, h_dot_B

    def CBF2_sim_loss(self,Fx,Tx):
        h = np.linalg.norm( Fx[0:2,0] - Tx[:,0] )**2 - self.min_D**2
        dh_dx = np.append(2*( Fx[0:2,0] - Tx[:,0] ),0)
        return dh_dx

    def CBF3_sim_loss(self,Fx,Tx):
        theta = Fx[2,0]
        # Direction Vector
        bearing_vector = np.array([np.cos(theta),np.sin(theta)]) @ (Tx[:,0] - Fx[0:2,0])/np.linalg.norm(Tx[:,0] - Fx[0:2,0])
        h = bearing_vector - np.cos(self.FoV_angle/2)

        dir_vector = np.array([np.cos(theta),np.sin(theta)])
        
        p_transpose = np.array([Tx[0,0],Tx[1,0]]) - np.array([Fx[0,0],Fx[1,0]])
        p = p_transpose.reshape(-1,1)
        factor = dir_vector/np.linalg.norm(Fx[0:2]-Tx[:,0]) - ( (dir_vector @ p) * p_transpose )/2/np.linalg.norm(Fx[0:2,0]-Tx[:,0])/(np.linalg.norm(Fx[0:2,0]-Tx[:,0])**2)

        factor_2 = ( (Tx[:,0] - Fx[0:2,0])/np.linalg.norm(Tx[:,0] - Fx[0:2,0]) @ np.array([ [-np.sin(theta)],[np.cos(theta)] ]) )

        dh_dx = np.append(-factor + factor_2,0)

        return dh_dx


    def CBF3_loss(self,target):
        theta = self.X[2,0]
        # Direction Vector
        bearing_vector = np.array([np.cos(theta),np.sin(theta)]) @ (target.X[:,0] - self.X[0:2,0])/np.linalg.norm(target.X[:,0] - self.X[0:2,0])
        h = (bearing_vector - np.cos(self.FoV_angle/2))/(1.0-np.cos(self.FoV_angle/2))

        dir_vector = np.array([np.cos(theta),np.sin(theta)])
        
        p_transpose = np.array([target.X[0,0],target.X[1,0]]) - np.array([self.X[0,0],self.X[1,0]])
        p = p_transpose.reshape(-1,1)
        factor = dir_vector/np.linalg.norm(self.X[0:2]-target.X[:,0]) - ( (dir_vector @ p) * p_transpose )/2/np.linalg.norm(self.X[0:2,0]-target.X[:,0])/(np.linalg.norm(self.X[0:2,0]-target.X[:,0])**2)
        factor_2 = ( (target.X[:,0] - self.X[0:2,0])/np.linalg.norm(target.X[:,0] - self.X[0:2,0]) @ np.array([ [-np.sin(theta)],[np.cos(theta)] ]) )

        # x = X_L - X_F
        dh_dx = dir_vector/np.linalg.norm(p,2) - ( (dir_vector @ p) * p_transpose )/np.linalg.norm(p)**3       
        dh_dTheta = np.array([ -np.sin(theta),np.cos(theta) ]) @ p/np.linalg.norm(p)

        dh_dx_agent = np.append(-dh_dx, dh_dTheta  )
        dh_dx_target = dh_dx
        # print(f"dh_dx:{dh_dx}, dh_dtheta:{dh_dTheta}, dh_dAgent:{dh_dx_agent}")

        # dh_dx_agent = np.append(-factor + factor_2, dh_dTheta  )
        # dh_dx_target = factor
        # vect = (target.X[:,0] - self.X[0:2,0])/np.linalg.norm(target.X[:,0] - self.X[0:2,0])
        # print("vect",vect)
        # dh_dTheta = np.array([-np.sin(theta),np.cos(theta)]) @ (target.X[:,0] - self.X[0:2,0])/np.linalg.norm(target.X[:,0] - self.X[0:2,0])
        # dh_dx_agent = np.append(-factor + factor_2, dh_dTheta  )
        # dh_dx_target = factor

        # h_dot_B = factor @ target.V.reshape(-1,1) - factor @ self.f[0:2,:] - factor @ self.f_corrected[0:2,:] + factor_2 * self.f[2,:] + factor_2 * self.f_corrected[2,:]
        # h_dot_A = -factor @ self.g[0:2,:] - factor @ self.g_corrected[0:2,:] + factor_2 * self.g[2,:] + factor_2 * self.g_corrected[2,:]
        return h, dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)), dh_dx_target/(1.0-np.cos(self.FoV_angle/2)) #h_dot_A, h_dot_B

    def CLF_loss(self,target):
        
        avg_D = (self.min_D + self.max_D)/2.0
        V = (np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)**2

        # V_dot = A*U + B
        factor = 2*(np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)/np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) * ( np.array([ self.X[0,0], self.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
        dV_dx_agent = [factor[0], factor[1], 0]
        dV_dx_target = -factor
        
        # V_dot_B = factor @ self.f[0:2,:] + factor @ self.f_corrected[0:2,:]
        # V_dot_A = factor @ self.g[0:2,:] + factor @ self.g_corrected[0:2,:]

        return V, dV_dx_agent, dV_dx_target #V_dot_A, V_dot_B

    def nominal_controller_old(self,target):
        #simple controller for now: does not consider disturbance

        #Define gamma for the Lyapunov function
        k_omega = 2.5
        k_v = 0.5

        theta_d = np.arctan2(target.X[:,0][1]-self.X[1,0],target.X[:,0][0]-self.X[0,0])
        error_theta = self.wrap_angle( theta_d - self.X[2,0] )
        omega = k_omega*error_theta

        distance = max(np.linalg.norm( self.X[0:2,0]-target.X[:,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta )**2 
        return v, omega 

    def nominal_controller_exact(self,target):
        #simple controller for now: considers true disturbance

        #Define gamma for the Lyapunov function
        k_omega = 2.5
        k_v = 0.5

        theta_d = np.arctan2(target.X[:,0][1]-self.X[1,0],target.X[:,0][0]-self.X[0,0])
        error_theta = self.wrap_angle( theta_d - self.X[2,0] )
        omega = k_omega*error_theta - self.d3

        distance = max(np.linalg.norm( self.X[0:2,0]-target.X[:,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta )**2 - (self.d1 + self.d2)/2
        return v, omega 

    def nominal_controller(self,target):
        #simple controller for now: considers estimated disturbance

        #Define gamma for the Lyapunov function
        k_omega = 0.0#2.5
        k_v = 0.5

        theta_d = np.arctan2(target.X[:,0][1]-self.X[1,0],target.X[:,0][0]-self.X[0,0])
        error_theta = self.wrap_angle( theta_d - self.X[2,0] )

        if (np.abs(1 + self.g_corrected[2,1])>0.01):
            omega = k_omega*error_theta/(1 + self.g_corrected[2,1]) - self.f_corrected[2,0]
        else:
            omega = k_omega*error_theta - self.f_corrected[2,0]

        distance = max(np.linalg.norm( self.X[0:2,0]-target.X[:,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta ) - (self.f_corrected[0,0] + self.f_corrected[1,0])/2
        return v, omega #np.array([v,omega])

    def compute_reward(self,target):

        # Want h positive
        h1,dh1_dxA,dh1_dxB = self.CBF1_loss(target)
        h2,dh2_dxA,dh2_dxB = self.CBF2_loss(target)
        h3,dh3_dxA,dh3_dxB = self.CBF3_loss(target)

        # find log sum maximum of negative h function -> minimum h function
        rho = 10.0
        barrier = 1/rho*np.log( np.exp(-rho*h1) + np.exp(-rho*h2) + np.exp(-rho*h3))

        h_list = [h1,h2,h3]
        dh_dx_list = [dh1_dxA,dh2_dxA,dh3_dxA]
        # barrier max min
        barrier = min(h_list)
        # print("reward inside", barrier)
        self.h_index = h_list.index(min(h_list))

        # if h1<0 or h2<0 or h3<0:
        #     return -1
        if barrier < 0:
            return barrier, 0, h1, h2, h3
            # return -1, 0
        
        return barrier, dh_dx_list[self.h_index], h1, h2, h3

    def compute_reward_sim(self,Fx,Tx):
        if self.h_index == 0:
            return self.CBF1_sim_loss(Fx,Tx)
        elif self.h_index == 1:
            return self.CBF2_sim_loss(Fx,Tx)
        else:    # self.h_index == 2:
            return self.CBF3_sim_loss(Fx,Tx)


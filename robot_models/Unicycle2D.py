import numpy as np
import cvxpy as cp
import torch
import math
import matplotlib.patches as mpatches

dt = 0.01

def fx_(x):
    return x + torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1)*dt,dtype=torch.float)
    # return torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1)*dt,dtype=torch.float)

def gx_(x):
    # x_ = x.detach().numpy()
    g_ = torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])*dt
    return g_ #torch.tensor(g_,dtype=torch.float)

def df_dx_(x, type = 'tensor'):
    dfdx = np.eye(3) + np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])*dt
    if type == 'numpy':
        return dfdx
    if type == 'tensor':
        return torch.tensor(dfdx,dtype=torch.float)
    
def dgxu_dx_(x, u, type='tensor'):
    # print(u)
    # u_ = u.detach().numpy()
    # dgxudx = np.array([[0.0, 0.0, -u[0,0]*np.sin(x[2,0]) ],[0.0, 0.0, u[0,0]*np.cos(x[2,0])],[ 0.0,0.0,0.0 ]])*dt
    dgxudx = torch.tensor([[0.0, 0.0, -u[0,0]*torch.sin(x[2,0]) ],[0.0, 0.0, u[0,0]*torch.cos(x[2,0])],[ 0.0,0.0,0.0 ]])*dt
    if type == 'numpy':
        return dgxudx.detach().numpy()
    if type == 'tensor':
        return dgxudx

class Unicycle2D:
    
    def __init__(self,X0,dt,fov_length,fov_angle,max_D,min_D):
        X0 = X0.reshape(-1,1)
        self.X  = X0
        self.dt = dt
        self.d1 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16) 
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        self.f = np.array([0.0,0.0,0.0]).reshape(-1,1)
        self.g = np.array([ [np.cos(X0[2,0]),0.0],[np.sin(X0[2,0]),0.0],[0.0,1.0] ])
        self.f_corrected = np.array([0.0,0.0,0.0]).reshape(-1,1)
        self.g_corrected = np.array([ [0.0,0.0],[0.0,0.0],[0.0,0.0] ])

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
        # self.d1 = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        # self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        # self.d3 = 0.3
        
        extra_f = np.array([self.d1*np.cos(self.X[2][0]),self.d2*np.sin(self.X[2][0]),self.d3]).reshape(-1,1)
        extra_g = np.array([ [0,0],[0,0],[0,0] ])
        self.X = self.X + (extra_f + extra_g @ U + self.f + self.g @ U)*self.dt
        # print("gU",self.g @ U)
    
        self.X[2,0] = self.wrap_angle(self.X[2,0])
        return self.X

    def fx(self,x):
        return torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1)*dt,dtype=torch.float)
    # return torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1)*dt,dtype=torch.float)

    def gx(self,x):
        return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])

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

    def arc_points(self, center, radius, theta1, theta2, resolution=50):
        # generate the points
        theta = np.linspace(theta1, theta2, resolution)
        points = np.vstack((radius*np.cos(theta) + center[0], 
                            radius*np.sin(theta) + center[1]))
        return points.T
    
    def render(self,lines,areas,body, poly, des_point):
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

        des_dist = self.min_D + (self.max_D - self.min_D)/2
        des_x = np.array( [ self.X[0,0] + np.cos(theta)*des_dist, self.X[1,0] + np.sin(theta)*des_dist    ] )

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        
        triangle_v = [ x,P1,P2,x ]  

        # lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)

        # scatter plot update
        body.set_offsets([x[0],x[1]])
        des_point.set_offsets([des_x[0], des_x[1]])

        #Fov arc
        poly.set_xy(self.arc_points(x, self.FoV_length, theta2, theta1))

        return lines, areas, body, poly, des_point

    def wrap_angle(self,angle):
        if angle>np.pi:
            angle = angle - 2*np.pi
        if angle<-np.pi:
            angle = angle + 2*np.pi
        return angle

    def xdot(self,U):
        # print("g",self.g + self.g_corrected)
        return ( self.f + self.f_corrected + (self.g + self.g_corrected) @ U )

    def xdot_cp(self,x,U):
        # print("g",self.g + self.g_corrected)
        return ( self.f + self.g @ U )

    def xdot_tensor(self,x,U):
        # print("g",self.g + self.g_corrected)
        return ( self.f + self.g @ U )
    
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
        ratio = self.max_D**2 - self.min_D**2
        dh_dx_agent = np.append(- 2*( self.X[0:2,0] - target.X[:,0] ),0)
        dh_dx_target = 2*( self.X[0:2,0] - target.X[:,0] )
        # print(f"1: {self.max_D}, x:{self.X}, tX:{target.X}, {np.linalg.norm( self.X[0:2,0] - target.X[:,0] )}, diff:{self.X[0:2,0] - target.X[:,0]}")
        return h/ratio, dh_dx_agent/ratio, dh_dx_target/ratio  #h_dot_A, h_dot_B

    def CBF1_loss_cp(self, x, target):
        # h = self.max_D**2 - cp.square(cp.norm( x[0:2,0] - target.X[:,0] ))
        h = self.max_D - cp.norm( x[0:2,0] - target.X[:,0] )
        dh_dx_agent = cp.hstack([- 2*( x[0,0] - target.X[0,0] ), - 2*( x[0,0] - target.X[0,0] ),0])
        dh_dx_target = cp.hstack( [2*( x[0,0] - target.X[0,0] ), 2*( x[1,0] - target.X[1,0] )] )
        
        return h, dh_dx_agent, dh_dx_target  #h_dot_A, h_dot_B

    def CBF1_loss_tensor(self, x, target):
        targetX = torch.tensor(target.X,dtype=torch.float)

        # print("1loss ",torch.square(torch.norm( x[0:2] - targetX )))
        h = self.max_D**2 - torch.reshape(torch.square(torch.norm( x[0:2] - targetX )),(1,1))
        dh_dx_agent = torch.cat( ( -2*( x[0:2] - targetX ), torch.tensor([[0.0]]) ), 0).T
        dh_dx_target =  2*( x[0:2] - targetX).T
        # print(f"1: {self.max_D}, x:{x}, tX:{targetX}, {torch.norm( x[0:2,0] - targetX )}, diff:{x[0:2] - targetX}, diff2:{x[0:2,0] - targetX[:,0]}")
        
        return h, dh_dx_agent @ self.fx(x), dh_dx_agent @ self.gx(x), dh_dx_target  #h_dot_A, h_dot_B

    def CBF1_sim_loss(self,Fx,Tx):
        h = self.max_D**2 - np.linalg.norm( Fx[0:2,0] - Tx[:,0] )**2
        dh_dx_agent = np.append(- 2*( Fx[0:2,0] - Tx[:,0] ),0)
        return dh_dx

    def CBF2_loss(self,target):
        ratio = self.max_D**2 - self.min_D**2
        h = np.linalg.norm( self.X[0:2,0] - target.X[:,0] )**2 - self.min_D**2
        dh_dx_agent = np.append(2*( self.X[0:2,0] - target.X[:,0] ),0)
        dh_dx_target = - 2*( self.X[0:2,0] - target.X[:,0] )

        return h/ratio, dh_dx_agent/ratio, dh_dx_target/ratio #h_dot_A, h_dot_B

    def CBF2_loss_cp(self, x, target):
        h = cp.square(cp.norm( x[0:2,0] - target.X[:,0] )) - self.min_D**2
        dh_dx_agent = cp.hstack([ 2*( x[0,0] - target.X[0,0] ), 2*( x[0,0] - target.X[0,0] ), 0]) 
        dh_dx_target = cp.hstack([- 2*( x[0,0] - target.X[0,0]), - 2*( x[1,0] - target.X[1,0]) ])

        return h, dh_dx_agent, dh_dx_target #h_dot_A, h_dot_B

    def CBF2_loss_tensor(self, x, target):
        targetX = torch.tensor(target.X,dtype=torch.float)
        h = torch.reshape(torch.square(torch.norm( x[0:2] - targetX )),(1,1)) - self.min_D**2
        dh_dx_agent = torch.cat( ( 2*( x[0:2] - targetX ), torch.tensor([[0.0]]) ), 0).T
        dh_dx_target = - 2*( x[0:2] - targetX).T

        return h, dh_dx_agent @ self.fx(x), dh_dx_agent @ self.gx(x), dh_dx_target #h_dot_A, h_dot_B

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

    def cp_cos(self,x):        
        cosine = 1 - cp.power(x,2)/2 + cp.power(x,4)/math.factorial(3) - cp.power(x,6)/math.factorial(6) + cp.power(x,8)/math.factorial(8) - cp.power(x,10)/math.factorial(10) + cp.power(x,12)/math.factorial(12)
        return cosine

    def cp_sin(self,x):        
        sine = x - cp.power(x,3)/6 + cp.power(x,5)/math.factorial(5) - cp.power(x,7)/math.factorial(7) + cp.power(x,9)/math.factorial(9) - cp.power(x,11)/math.factorial(11) + cp.power(x,13)/math.factorial(13)
        return sine


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

        return h, dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)), dh_dx_target/(1.0-np.cos(self.FoV_angle/2)) #h_dot_A, h_dot_B

    def CBF3_loss_cp(self,x,cos_sin,target):
       
        print("diff", target.X - x[0:2])
        bearing_vector  = cos_sin.T @ (target.X - x[0:2])
        bearing_vector = bearing_vector/cp.norm(target.X[:,0] - x[0:2,0])

        h = (bearing_vector - np.cos(self.FoV_angle/2))/(1.0-np.cos(self.FoV_angle/2))

        dir_vector = cos_sin
        
        # p_transpose = np.array([target.X[0,0],target.X[1,0]]) - np.array([x[0,0],x[1,0]])
        # p = p_transpose.reshape(-1,1)
        p = target.X - x[0:2]
        norm_p = cp.norm(p)
        # x = X_L - X_F
        #dh_dx = dir_vector/cp.norm(p,2) - ( (dir_vector @ p) * p_transpose )/cp.norm(p)**3   
        print("1",dir_vector/norm_p)
        dh_dx = dir_vector/norm_p - ( ( dir_vector.T @ p ) * p )/cp.power(norm_p,3)
        print("dhdx",dh_dx.shape)
            
        dh_dTheta = ( -sin_theta * p[0,0] + cos_theta * p[1,0] )/cp.norm(p)
        dh_dx_agent = cp.hstack([-dh_dx[0,0], -dh_dx[1,0], dh_dTheta ])
        dh_dx_target = cp.hstack([dh_dx[0,0], dh_dx[1,0] ])
        # print(f"dh_dx:{dh_dx}, dh_dtheta:{dh_dTheta}, dh_dAgent:{dh_dx_agent}")

        return h, dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)), dh_dx_target/(1.0-np.cos(self.FoV_angle/2)) #h_dot_A, h_dot_B

    def CBF3_loss_tensor(self,x,target):
           
        targetX = torch.tensor(target.X,dtype=torch.float)
        p = targetX - x[0:2]

        dir_vector = torch.tensor([[torch.cos(x[2,0])],[torch.sin(x[2,0])]]) # column vector
        bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
        h = (bearing_angle - np.cos(self.FoV_angle/2))/(1.0-np.cos(self.FoV_angle/2))

        norm_p = torch.norm(p)
        dh_dx = dir_vector/norm_p - ( ( torch.matmul(dir_vector.T , p) ) * p )/torch.pow(norm_p,3)    
        dh_dTheta = torch.reshape( ( -torch.sin(x[2]) * p[0] + torch.cos(x[2]) * p[1] ),(1,1) )/torch.norm(p)
        dh_dx_agent = torch.cat(  ( -dh_dx , dh_dTheta),0  ).T
        dh_dx_target = dh_dx.T

        return h, dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)) @ self.fx(x), dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)) @ self.gx(x), dh_dx_target/(1.0-np.cos(self.FoV_angle/2)) #h_dot_A, h_dot_B


    def CBF3_loss_cp_v2(self,x,target):
        # print(type(x))
        theta = x[2,0]
        sin_theta = self.cp_sin(theta)
        cos_theta = self.cp_cos(theta)
        # Direction Vector
        # bearing_vector = np.array([self.cp_cos(theta),self.cp_sin(theta)]) @ (target.X[:,0] - x[0:2,0])/cp.norm(target.X[:,0] - x[0:2,0])

        bearing_vector  = cos_theta * (target.X[0,0] - x[0,0]) + sin_theta * (target.X[1,0] - x[1,0])
        bearing_vector = bearing_vector/cp.norm(target.X[:,0] - x[0:2,0])

        h = (bearing_vector - np.cos(self.FoV_angle/2))/(1.0-np.cos(self.FoV_angle/2))

        dir_vector = cp.vstack([cos_theta,sin_theta])
        
        # p_transpose = np.array([target.X[0,0],target.X[1,0]]) - np.array([x[0,0],x[1,0]])
        # p = p_transpose.reshape(-1,1)
        p = cp.vstack([target.X[0,0]-x[0,0], target.X[1,0]-x[1,0] ])
        norm_p = cp.sqrt( cp.square(p[0]) + cp.square(p[1]) )
        # x = X_L - X_F
        #dh_dx = dir_vector/cp.norm(p,2) - ( (dir_vector @ p) * p_transpose )/cp.norm(p)**3   
        print("1",dir_vector/norm_p)
        dh_dx = dir_vector/norm_p - ( (dir_vector[0] * p[0,0] + dir_vector[1] * p[1,0]) * p )/cp.power(norm_p,3)
        print("dhdx",dh_dx.shape)
            
        dh_dTheta = ( -sin_theta * p[0,0] + cos_theta * p[1,0] )/cp.norm(p)
        dh_dx_agent = cp.hstack([-dh_dx[0,0], -dh_dx[1,0], dh_dTheta ])
        dh_dx_target = cp.hstack([dh_dx[0,0], dh_dx[1,0] ])
        # print(f"dh_dx:{dh_dx}, dh_dtheta:{dh_dTheta}, dh_dAgent:{dh_dx_agent}")

        return h, dh_dx_agent/(1.0-np.cos(self.FoV_angle/2)), dh_dx_target/(1.0-np.cos(self.FoV_angle/2)) #h_dot_A, h_dot_B



    def CLF_loss(self,target):
        
        avg_D = (self.min_D + self.max_D)/2.0
        V = (np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)**2

        # V_dot = A*U + B
        factor = 2*(np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)/np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) * ( np.array([ self.X[0,0], self.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
        dV_dx_agent = [factor[0], factor[1], 0]
        dV_dx_target = -factor

        return V, dV_dx_agent, dV_dx_target #V_dot_A, V_dot_B

    def CLF_loss_tensor(self,x,target):
        
        targetX = torch.tensor(target.X,dtype=torch.float)
        avg_D = (self.min_D + self.max_D)/2.0
        V = torch.reshape(torch.pow(torch.norm(targetX)- avg_D,2),(1,1))

        p = targetX - x[0:2]
        factor = 2*(torch.norm( p ) - avg_D)/torch.norm(p) * ( -p )
        dV_dx_agent = torch.cat( ( factor , torch.tensor([[0]]))  , 0 ).T
        dV_dx_target = -factor.T

        return V, dV_dx_agent @ self.fx(x), dV_dx_agent @ self.gx(x), dV_dx_target 

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

    def nominal_controller_cp(self,x, target):
        #simple controller for now: considers estimated disturbance
        #Define gamma for the Lyapunov function
        k_omega = 0.0#2.5
        k_v = 0.5

        theta_d = np.arctan2(target.X[:,0][1]-x[1,0],target.X[:,0][0]-x[0,0])
        error_theta = self.wrap_angle( theta_d - x[2,0] )

        omega = k_omega*error_theta 

        distance = max(np.linalg.norm( x[0:2,0]-target.X[:,0] ) - 0.3,0)
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

    def compute_reward_tensor(self,x,target):
        targetX = torch.tensor(target.X,dtype=torch.float)
        diff = targetX[:,0] - x[0:2,0]  # row vector

        # Max distance
        h1 = (self.max_D**2 - torch.norm( diff )**2)/(self.max_D**2-self.min_D**2)

        # Min distance
        h2 = (torch.norm( diff )**2 - self.min_D**2)/(self.max_D**2-self.min_D**2)
        # print(f"reward x:{x}, shape:{x.shape}")
        # Max angle
        bearing_vector = torch.tensor( [[torch.cos(x[2,0]), torch.sin(x[2,0])]] , dtype=torch.float,requires_grad=True)
        bearing_angle = torch.matmul ( bearing_vector , diff ) /torch.norm(diff)
        h3 = (bearing_angle - np.cos(self.FoV_angle/2))[0]/(1.0-np.cos(self.FoV_angle/2))
        # reward is minimum reward
        return torch.minimum(torch.minimum(h1,h2),h3)

    def compute_reward_sim(self,Fx,Tx):
        if self.h_index == 0:
            return self.CBF1_sim_loss(Fx,Tx)
        elif self.h_index == 1:
            return self.CBF2_sim_loss(Fx,Tx)
        else:    # self.h_index == 2:
            return self.CBF3_sim_loss(Fx,Tx)


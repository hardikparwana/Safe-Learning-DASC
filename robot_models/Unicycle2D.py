import numpy as np
import cvxpy as cp

class Unicycle2D:
    
    def __init__(self,X0,dt,fov_length,fov_angle,max_D,min_D):
        X0 = X0.reshape(-1,1)
        self.X  = X0
        self.dt = dt

        self.f = np.array([0.0,0.0,0.0]).reshape(-1,1)
        self.g = np.array([ [np.cos(X0[2,0]),0.0],[np.sin(X0[2,0]),0.0],[0.0,1.0] ])

        self.FoV_length = fov_length
        self.FoV_angle = fov_angle

        self.max_D = max_D
        self.min_D = min_D

        self.h_index = 1
        
    def step(self,U):
        U = U.reshape(-1,1)
        self.g = np.array([ [np.cos(self.X[2,0]),0],[np.sin(self.X[2,0]),0],[0,1] ])

        self.X = self.X + (self.f + self.g @ U)*self.dt
    
        self.X[2,0] = self.wrap_angle(self.X[2,0])
        return self.X

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


    def CBF2_loss(self,target):
        ratio = self.max_D**2 - self.min_D**2
        h = np.linalg.norm( self.X[0:2,0] - target.X[:,0] )**2 - self.min_D**2
        dh_dx_agent = np.append(2*( self.X[0:2,0] - target.X[:,0] ),0)
        dh_dx_target = - 2*( self.X[0:2,0] - target.X[:,0] )

        return h/ratio, dh_dx_agent/ratio, dh_dx_target/ratio #h_dot_A, h_dot_B


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

    def CLF_loss(self,target):
        
        avg_D = (self.min_D + self.max_D)/2.0
        V = (np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)**2

        # V_dot = A*U + B
        factor = 2*(np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) - avg_D)/np.linalg.norm( self.X[0:2,0] - target.X[:,0] ) * ( np.array([ self.X[0,0], self.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
        dV_dx_agent = [factor[0], factor[1], 0]
        dV_dx_target = -factor

        return V, dV_dx_agent, dV_dx_target #V_dot_A, V_dot_B

    def nominal_controller(self,target):
        #simple controller for now: considers estimated disturbance

        #Define gamma for the Lyapunov function
        k_omega = 0.0#2.5
        k_v = 0.5

        theta_d = np.arctan2(target.X[:,0][1]-self.X[1,0],target.X[:,0][0]-self.X[0,0])
        error_theta = self.wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta

        distance = max(np.linalg.norm( self.X[0:2,0]-target.X[:,0] ) - 0.3,0)
        v = k_v*( distance )*np.cos( error_theta )

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
       
        self.h_index = h_list.index(min(h_list))

        # if h1<0 or h2<0 or h3<0:
        #     return -1
        if barrier < 0:
            return barrier, 0, h1, h2, h3
        
        return barrier, dh_dx_list[self.h_index], h1, h2, h3



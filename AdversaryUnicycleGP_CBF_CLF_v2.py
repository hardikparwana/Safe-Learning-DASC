import numpy as np
import random
import matplotlib.pyplot as plt

import os
import time

from cvxopt import matrix
from cvxopt import solvers

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C




from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
# from sklearn.exceptions import UserWarning

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

def true_d(theta,index):
    if index==0:
        return 1*np.sin(theta)**2 + 1*np.sin(theta*16)
    elif index==1:
        return 1*np.sin(theta)**2 + 1*np.sin(theta*16)
    else:
        return theta*0+0.3

class follower:
    
    def __init__(self,X0,dt):
        X0 = X0.reshape(-1,1)
        self.X  = X0
        self.dt = dt
        self.d1 = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        self.f = np.array([0,0,0]).reshape(-1,1)
        self.g = np.array([ [np.cos(X0[2,0]),0],[np.sin(X0[2,0]),0],[0,1] ])
        self.f_corrected = np.array([0,0,0]).reshape(-1,1)
        self.g_corrected = np.array([ [0,0],[0,0],[0,0] ])
        
    def step(self,U):
        U = U.reshape(-1,1)
        self.g = np.array([ [np.cos(self.X[2,0]),0],[np.sin(self.X[2,0]),0],[0,1] ])
        # print("agent G",self.g)
        # disturbance term
        self.d1 = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        
        extra_f = np.array([self.d1*np.cos(self.X[2][0]),self.d2*np.sin(self.X[2][0]),self.d3]).reshape(-1,1)
        extra_g = np.array([ [0,0],[0,0],[0,0] ])
        # print(f"true d1:{self.d1} X: {self.X} , f:{d1*np.cos(self.X[2])}, g:{u*np.cos(self.X[2])}, cos term: {np.cos(self.X[2])} ")
        # print(f" extra_f:{extra_f}, extra_g:{extra_g}, self.f:{self.f}, self.g:{self.g}, U;{U} ")
        # print("X old",self.X)
        
        self.X = self.X + (extra_f + extra_g @ U + self.f + self.g @ U)*self.dt

        # print(f"step: X_next:{self.X}")
        # print("X",self.X)
    
        self.X[2] = wrap_angle(self.X[2])
        
        return self.X

    def step_sim(self,u,w):
        
        f = np.array([u*np.cos(self.X[2]),u*np.sin(self.X[2]),w])
        X_sim = self.X + f*self.dt
        X_sim[2] = wrap_angle(X_sim[2])
        
        return X_sim
    
    def render(self,lines,areas,body):
        length = 3
        # FoV = np.pi/3   # 60 degrees

        x = np.array([self.X[0,0],self.X[1,0]])
        # print("X",self.X)
        theta = self.X[2][0]
        theta1 = theta + FoV/2
        theta2 = theta - FoV/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + length*e1
        P2 = x + length*e2  

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        
        triangle_v = [ x,P1,P2,x ]  
        # print("triangle_v",triangle_v)

        lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)

        length2 = 3

        # scatter plot update
        body.set_offsets([x[0],x[1]])
#         sc.set_offsets(np.c_[x,y])

        return lines, areas, body
       
class target:
    
    def __init__(self,X0,dt):
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.V = np.array([0,0]).reshape(-1,1)
        self.dt = dt
        self.t0 = 0
        self.speed = 0
        self.theta = 0
        
    def step(self,a,alpha): #Just holonomic X,T acceleration
        
        if (self.speed<2):
            self.speed = self.speed + a*self.dt

        self.V[0,0] = a
        self.V[1,0] = alpha

        self.X = self.X + np.array([a,alpha]).reshape(-1,1)*dt
        return self.X

    def step_sim(self,a,alpha): #Just holonomic X,T acceleration
        
        X_sim = self.X + np.array([a,alpha])*dt
        return X_sim
    
    def render(self,body):
        length = 3
        # FoV = np.pi/3

        x = np.array([self.X[0,0],self.X[1,0]])

        # scatter plot update
        body.set_offsets([x[0],x[1]])

        return body
    
def wrap_angle(angle):
    if angle>np.pi:
        angle = angle - 2*np.pi
    if angle<-np.pi:
        angle = angle + 2*np.pi
    return angle


def build_GP_model(N):
    GP_list = []
    noise = 0.01
    for i in range(N):
        kern = 1.0 * RBF(length_scale=2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
        #kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
        # gp = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)
        GP_list.append(gp)
    return GP_list

@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ConvergenceWarning)
def update_GP_dynamics(GP_list,X,Y, index, curr_X):
    # X, Y = filter_data(X,Y,curr_X)
    # print("start")
    GP_list[index].fit(np.asarray(X).reshape(-1,1),np.asarray(Y).reshape(-1,1))
    # print("done")


def filter_data(obs_x,obs_value,curr_x,thres=0.5):
    filter_x = []
    filter_value = []
    for index, t in enumerate(obs_x):
        if np.abs(t-curr_x)<thres:
            filter_x.append(t)
            filter_value.append(obs_value[index])
    return filter_x, filter_value

def predict_GP_dynamics(GP_list,N,X):
    Y = []
    Y_std = []
    for i in range(N):
        y_pred, sigma_pred = GP_list[i].predict(np.array([X]).reshape(-1, 1),return_std=True)
        # print(f"y:{y_pred[0][0]}, y_pred: {sigma_pred[0]}")
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def test_GP_dynamics(GP_list,X,index):
    Y = []
    Y_std = []
    for i in X:
        y_pred, sigma_pred = GP_list[index].predict(np.array([i]).reshape(-1, 1),return_std=True)
        # print(f"y:{y_pred[0][0]}, y_pred: {sigma_pred[0]}")
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def compute_CLF(X,Xd):
    V = (np.linalg.norm(X-Xd))*2
    return V



def CBF1_loss(agent,target):
    h = max_D**2 - np.linalg.norm( agent.X[0:2,0] - target.X[:,0] )**2
    # print("h1",h)
    # print(f"F_X:{agent.X}, T_X:{target.X}")
    # print("distance",np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ))
    # h_dot = 2*( agent.X[0:2,0] - target.X[:,0] )*target.V  - 2*( agent.X[0:2,0] - target.X[:,0] )*( agent.f[0:2,:] + agent.g[0:2,:] @ U  ) - 2*( agent.X[0:2,0] - target.X[:,0] )*( agent.f_corrected + agent.g_corrected[0:2,:] @ U  )
    
    # h_dot = A*U + B
    # print("1: ", agent.X[0:2,0] - target.X[:,0][:,0])
    # print("2: ",agent.g[0:2,:])
    h_dot_B = 2*( agent.X[0:2,0] - target.X[:,0] ) @ target.V  - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f[0:2,:]  ) - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f_corrected[0:2,:]  )
    h_dot_A = - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g[0:2,:]  ) - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g_corrected[0:2,:]  )
    
    return h, h_dot_A, h_dot_B

def CBF2_loss(agent,target):
    h = np.linalg.norm( agent.X[0:2,0] - target.X[:,0] )**2 - min_D**2
    # print("h2",h)
    # h_dot = -2*( agent.X[0:2,0] - target.X[:,0] )*target.V + 2*( agent.X[0:2,0] - target.X[:,0] )*( agent.f[0:2,:] + agent.g[0:2,:] @ U  ) + 2*( agent.X[0:2,0] - target.X[:,0] )*( agent.f_corrected + agent.g_corrected[0:2,:] @ U  )
    
    h_dot_B = - 2*( agent.X[0:2,0] - target.X[:,0] ) @ target.V  + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f[0:2,:]  ) + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f_corrected[0:2,:]  )
    h_dot_A =   2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g[0:2,:]  ) + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g_corrected[0:2,:]  )
    # print("1: ", 2*( agent.X[0:2,0] - target.X[:,0][:,0] ) @ ( agent.g[0:2,:]  ))
    # print("2: ", 2*( agent.X[0:2,0] - target.X[:,0][:,0] ) @ ( agent.g_corrected[0:2,:]  ))
    return h, h_dot_A, h_dot_B


def CBF3_loss(agent,target):
    theta = agent.X[2,0]
    # Direction Vector
    bearing_vector = np.array([np.cos(theta),np.sin(theta)]) @ (target.X[:,0] - agent.X[0:2,0])/np.linalg.norm(target.X[:,0] - agent.X[0:2,0])
    h = bearing_vector - np.cos(FoV/2)

    # print("h3",h)

    dir_vector = np.array([np.cos(theta),np.sin(theta)])
    
    p_transpose = np.array([target.X[0,0],target.X[1,0]]) - np.array([agent.X[0,0],agent.X[1,0]])
    p = p_transpose.reshape(-1,1)
    # print("first", dir_vector/np.linalg.norm(agentF.X[0:2]-target.X[:,0])  )
    # print("second", (dir_vector @ p)* p_transpose)
    factor = dir_vector/np.linalg.norm(agentF.X[0:2]-target.X[:,0]) - ( (dir_vector @ p) * p_transpose )/2/np.linalg.norm(agent.X[0:2,0]-target.X[:,0])/(np.linalg.norm(agent.X[0:2,0]-target.X[:,0])**2)
    # actual = agent.f[0:2,:] + agent.g[0:2,:] @ U
    # estimated = agent.f_corrected[0:2,:] + agent.g_corrected[0:2,:] @ U
    # h_dot = factor @ target.V.reshape(-1,1)  - factor @ actual

    factor_2 = ( (target.X[:,0] - agent.X[0:2,0])/np.linalg.norm(target.X[:,0] - agent.X[0:2,0]) @ np.array([ [-np.sin(theta)],[np.cos(theta)] ]) )
    # print("factor2",factor_2)
    h_dot_B = factor @ target.V.reshape(-1,1) - factor @ agent.f[0:2,:] - factor @ agent.f_corrected[0:2,:] + factor_2 * agent.f[2,:] + factor_2 * agent.f_corrected[2,:]
    # print("1: ",factor @ agent.g[0:2,:])
    # print("2: ",factor @ agent.g_corrected[0:2,:])
    # print("3: ",factor_2 * agent.g[2,:])

    h_dot_A = -factor @ agent.g[0:2,:] - factor @ agent.g_corrected[0:2,:] + factor_2 * agent.g[2,:] + factor_2 * agent.g_corrected[2,:]
    return h, h_dot_A, h_dot_B

def CLF_loss(agent,target):
    
    avg_D = (min_D + max_D)/2.0
    V = (np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) - avg_D)**2
    # print("Laypunov",V)

    # V_dot = A*U + B
    factor = (np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) - avg_D)/np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) * 2 * ( np.array([ agent.X[0,0], agent.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
    V_dot_B = factor @ agent.f[0:2,:] + factor @ agent.f_corrected[0:2,:]
    V_dot_A = factor @ agent.g[0:2,:] + factor @ agent.g_corrected[0:2,:]

    return V, V_dot_A, V_dot_B



# def nominal_controller_old(X,Xd):#,d1,d2,d3):
#     #simple controller for now

#     #Define gamma for the Lyapunov function
#     k_omega = 2.5
#     k_v = 0.5

#     theta_d = np.arctan2(Xd[1]-X[1],Xd[0]-X[0])
#     error_theta = wrap_angle( theta_d - X[2] )
#     omega = k_omega*error_theta# - d2

#     distance = max(np.linalg.norm( X[0:2]-Xd ) - 0.3,0)

#     v = k_v*( distance )*np.cos( error_theta )**2# - (d1 + d2)/2
#     # print(f"w:{omega}, v:{v}")
#     return v, omega #np.array([v,omega])

def nominal_controller_old(agent,target):
    #simple controller for now

    #Define gamma for the Lyapunov function
    k_omega = 2.5
    k_v = 0.5

    theta_d = np.arctan2(target.X[:,0][1]-agent.X[1,0],target.X[:,0][0]-agent.X[0,0])
    error_theta = wrap_angle( theta_d - agent.X[2,0] )
    omega = k_omega*error_theta

    distance = max(np.linalg.norm( agent.X[0:2,0]-target.X[:,0] ) - 0.3,0)

    v = k_v*( distance )*np.cos( error_theta )**2 
    # print(f"w:{omega}, v:{v}")
    return v, omega 

def nominal_controller_exact(agent,target):
    #simple controller for now

    #Define gamma for the Lyapunov function
    k_omega = 2.5
    k_v = 0.5

    theta_d = np.arctan2(target.X[:,0][1]-agent.X[1,0],target.X[:,0][0]-agent.X[0,0])
    error_theta = wrap_angle( theta_d - agent.X[2,0] )
    omega = k_omega*error_theta - agent.d3

    distance = max(np.linalg.norm( agent.X[0:2,0]-target.X[:,0] ) - 0.3,0)

    v = k_v*( distance )*np.cos( error_theta )**2 - (agent.d1 + agent.d2)/2
    # print(f"w:{omega}, v:{v}")
    return v, omega 

def nominal_controller(agent,target):
    #simple controller for now

    #Define gamma for the Lyapunov function
    k_omega = 2.5
    k_v = 0.5

    theta_d = np.arctan2(target.X[:,0][1]-agent.X[1,0],target.X[:,0][0]-agent.X[0,0])
    error_theta = wrap_angle( theta_d - agent.X[2,0] )

    if (np.abs(1 + agent.g_corrected[2,1])>0.01):
        omega = k_omega*error_theta/(1 + agent.g_corrected[2,1]) - agent.f_corrected[2,0]
    else:
        omega = k_omega*error_theta - agent.f_corrected[2,0]

    distance = max(np.linalg.norm( agent.X[0:2,0]-target.X[:,0] ) - 0.3,0)

    v = k_v*( distance )*np.cos( error_theta )**2 - (agent.f_corrected[0,0] + agent.f_corrected[1,0])/2
    # print(f"w:{omega}, v:{v}")
    return v, omega #np.array([v,omega])

def CBF_CLF_QP(agent,target):

    u, w = nominal_controller(agent,target)
    U = np.array([u,w]).reshape(-1,1)

    alpha = 0.1 #1.0 does not work
    k = 0.1

    h1,h1A,h1B = CBF1_loss(agent,target)
    
    h2,h2A,h2B = CBF2_loss(agent,target)
    # print(f"h1:{h1}, a:{h1A}, b:{h1B}")
    # print(f"h2:{h2}, a:{h2A}, b:{h2B}")
    h3,h3A,h3B = CBF3_loss(agent,target)

    V,VA,VB = CLF_loss(agent,target)

    G = np.array([ [ VA[0] , VA[1], -1.0   ],
                   [ -h1A[0], -h1A[1], 0.0 ] ,
                   [ -h2A[0], -h2A[1], 0.0 ] ,
                   [ -h3A[0], -h3A[1], 0.0 ]    
                    ])

    # G = np.array([ [ 0 , 0, 1.0   ],
    #                [ 2, 0, 0.0 ] ,
    #                [ -2, 0, 0.0 ] ,
    #                [ -0.77, -0.19611614, 0.0 ]    
    #                 ])


    
    h = np.array([-VB - k*V ,
                 h1B + alpha*h1  ,
                 h2B + alpha*h2  ,
                 h3B + alpha*h3 ]).reshape(-1,1)

    # print("G",G)
    # print("h",h)

    # print(f"u:{u}, w:{w}")
    #Convert numpy arrays to cvx matrices to set up QP
    G = matrix(G,tc='d')
    h = matrix(h,tc='d')

    # Cost matrices
    P = matrix(np.diag([2., 2., 6.]), tc='d')
    q = matrix(np.array([ -2*u, -2*w, 0 ]))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u_bar = sol['x']

    c_input = np.array([u_bar[0],u_bar[1]]).reshape(-1,1)
    # print("Constraint 1", h1B + h1A @ c_input + alpha * h1 )
    # print("Constraint 2", h2B + h2A @ c_input + alpha * h2 )
    # print("Constraint 3", h3B + h3A @ c_input + alpha * h3 )
    
    # print("u_bar",u_bar)
    # print("u_bar[0]",u_bar[0])
    # print("slack",u_bar[2])

    # exit()

    return u_bar[0], u_bar[1]






dt = 0.1 #0.05 #0.1
tf = 1 #20
N = 3

GP_list = build_GP_model(N)

agentF = follower(np.array([0,0.2,0]),dt)
agentT = target(np.array([1,0]),dt)

plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,20),ylim=(-10,10))

lines, = ax.plot([],[],'o-')
areas, = ax.fill([],[],'r',alpha=0.1)
bodyF = ax.scatter([],[],c='r',s=10)

bodyT = ax.scatter([],[],c='g',s=10)

t = 0

obs_d1 = []
obs_d2 = []
obs_d3 = []

true_d1 = []
true_d2 = []
true_d3 = []

pred_d1 = []
pred_d2 = []
pred_d3 = []

h1p = []
h2p = []
h3p = []

debug_value = []

Xgp = []

for k in range(0,60):

    uL = 0.5
    vL = 3.6*np.sin(np.pi*t) #  0.1 # 1.2

    if t==0:
        d1 = 0
        d2 = 0
        d3 = 0
    else:
        d, d_std = predict_GP_dynamics( GP_list,N,agentF.X[2]  )
        d1 = d[0]
        d2 = d[1]
        d3 = d[2]

    pred_d1.append(d1)
    pred_d2.append(d2)
    pred_d3.append(d3)

    agentF.f_corrected = np.array([agentF.d1*np.cos(agentF.X[2][0]),agentF.d2*np.sin(agentF.X[2][0]),agentF.d3]).reshape(-1,1)
    # predicted
    # agentF.f_corrected = np.array([d1*np.cos(agentF.X[2][0]),d2*np.sin(agentF.X[2][0]),d3]).reshape(-1,1)
    # ignored
    # agentF.f_corrected = np.array([0*np.cos(agentF.X[2][0]),0*np.sin(agentF.X[2][0]),0]).reshape(-1,1)
    agentF.g_corrected = np.array([ [0,0],[0,0],[0,0] ])

    Xgp.append(agentF.X[2,0])

    # u,w = nominal_controller_old(agentF,agentT) # no estimation
    # u,w = nominal_controller_exact(agentF,agentT) # known noise
    # u,w = nominal_controller(agentF,agentT) # using estimated dynamics
    u,w = CBF_CLF_QP(agentF,agentT) # using estimated dynamics

    h ,_ ,_ = CBF1_loss(agentF,agentT)
    h1p.append(h)
    h ,_ ,_ = CBF2_loss(agentF,agentT)
    h2p.append(h)
    h ,_ ,_ = CBF3_loss(agentF,agentT)
    h3p.append(h)

    # print("d1",d1)
    # print("d2",d2)

    FX_prev = agentF.X    

    agentT.step(uL,vL)  #agentT.step(0.2,0.5)  
    U = np.array([u,w]).reshape(-1,1)      
    agentF.step(U)

    true_d1.append(agentF.d1)#(0.3*np.sin(agentF.X[2])**2) #sin(u)
    true_d2.append(agentF.d2)#(0.3*np.sin(agentF.X[2])**2)
    true_d3.append(agentF.d3)#(0.3)

    FX = agentF.X
    # print("d1 from F",agentF.d1)
    if (np.abs(np.cos(FX_prev[2]))>0.01):
        d1_obs = ( (FX[0]-FX_prev[0])/dt - u*np.cos(FX_prev[2]) )/np.cos(FX_prev[2])
        # print("d1_obs",d1_obs)
        # print(f"d1_obs: {d1_obs}, X:{FX_prev}, X_next:{FX}, gu :{u*np.cos(FX_prev[2])}, f:{ (FX[0]-FX_prev[0])/dt - u*np.cos(FX_prev[2]) } ")
    else:
        # print("0000")
        d1_obs = 0
    if (np.abs(np.sin(FX_prev[2]))>0.01):
        d2_obs = ( (FX[1]-FX_prev[1])/dt - u*np.sin(FX_prev[2]) )/np.sin(FX_prev[2])
    else:
        d2_obs = 0
    d3_obs =  wrap_angle(FX[2]-FX_prev[2])/dt - w

    obs_d1.append(d1_obs)
    obs_d2.append(d2_obs)
    obs_d3.append(d3_obs)

    debug_value.append(FX_prev[2])    

    # update_GP_dynamics(GP_list,N,np.array([FX_prev[2],FX_prev[2],FX_prev[2]] ), np.array([d1_obs,d2_obs,d3_obs]))
    update_GP_dynamics(GP_list,Xgp, obs_d1, 0, agentF.X[2,0])
    update_GP_dynamics(GP_list,Xgp, obs_d2, 1, agentF.X[2,0])
    update_GP_dynamics(GP_list,Xgp, obs_d3, 2, agentF.X[2,0])

    # lines, areas, bodyF = agentF.render(lines,areas,bodyF)
    # bodyT = agentT.render(bodyT)
    
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    t = t + dt



print(len(pred_d1))
print(len(true_d1))

print(pred_d1)
print(true_d1)
print("DONE!!")
plt.ioff()


plt.figure()
theta = np.linspace(-1,2,100)
d, d_std = test_GP_dynamics( GP_list,theta, 0  )
d_true = true_d(theta,0)
plt.plot(theta,d_true,'r',label='True Dyanmics')
plt.plot(theta,d,label='Estimated Dynamics')
plt.legend()
plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')
plt.title("Supervised Unknown Dynamics Estimation")

plt.figure()
plt.plot(obs_d1,'g*',label='True Dynamics')
plt.plot(true_d1,'r',label='Training Points')
plt.plot(pred_d1,'c',label='Predicted Valaues')
plt.xlabel('time (s)')
plt.legend()
plt.title('Supervised Gaussian Process Prediction')

plt.show()


exit()

plt.figure()
theta = np.linspace(-1,1,100)
d, d_std = test_GP_dynamics( GP_list,theta, 0  )
d_true = true_d(theta,0)
plt.plot(theta,d_true,'r',label='True Dyanmics')
plt.plot(theta,d,label='Estimated Dynamics')
plt.legend()
plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')
plt.title("Unknown Dynamics Estimation")

plt.figure()
plt.subplot(3,1,1)
theta = np.linspace(-1,1,100)
d, d_std = test_GP_dynamics( GP_list,theta, 0  )
d_true = true_d(theta,0)
plt.plot(theta,d_true,'r')


# print("d1",d)
    
# print(f"d size:{d.size}, shape:{d.shape}, d:{d}, theta:{theta}")
# print("d_std:",d_std)
# print("d",d)
plt.plot(theta,d)
plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')
plt.title("d1")

plt.subplot(3,1,2)
theta = np.linspace(-1,1,100)
d, d_std = test_GP_dynamics( GP_list,theta, 1  )
d_true = true_d(theta,1)
plt.plot(theta,d_true,'r')
# print(f"d size:{d.size}, shape:{d.shape}, d:{d}, theta:{theta}")
# print("d_std:",d_std)
# print("d",d)
plt.plot(theta,d)
plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')
plt.title("d2")

plt.subplot(3,1,3)
theta = np.linspace(-1,1,100)
d, d_std = test_GP_dynamics( GP_list,theta, 2  )
d_true = true_d(theta,2)
plt.plot(theta,d_true,'r')
# print(f"d size:{d.size}, shape:{d.shape}, d:{d}, theta:{theta}")
# print("d_std:",d_std)
# print("d",d)
plt.plot(theta,d)
plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')
plt.title("d3")

# plt.figure()
# plt.plot(theta,d)

plt.figure()
plt.plot(obs_d1,'g*',label='True Dynamics')
plt.plot(true_d1,'r',label='Training Points')
plt.plot(pred_d1,'c',label='Predicted Valaues')
plt.xlabel('time (s)')
plt.legend()
plt.title('Gaussian Process Prediction')

plt.figure()

plt.subplot(3,1,1)
plt.plot(obs_d1,'g*')
plt.plot(true_d1,'r')
plt.plot(pred_d1,'c')

plt.subplot(3,1,2)
plt.plot(obs_d2,'g*')
plt.plot(true_d2,'r')
plt.plot(pred_d2,'c')

plt.subplot(3,1,3)
plt.plot(obs_d3,'g*')
plt.plot(true_d3,'r')
plt.plot(pred_d3,'c')

plt.figure()
plt.plot(debug_value,'k')


hor = np.zeros(len(h1p))
plt.figure()
plt.subplot(3,1,1)
plt.plot(h1p,'g')
plt.plot(hor,'k')

plt.subplot(3,1,2)
plt.plot(h2p,'g')
plt.plot(hor,'k')


plt.subplot(3,1,3)
plt.plot(h3p,'g')
plt.plot(hor,'k')



print("h1p",h1p)
print("h2p",h2p)
print("h3p",h3p)

plt.show()



    
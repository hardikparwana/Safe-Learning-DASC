'''
---------------------------------------------------------
Implementing of TD3
---------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import namedtuple, deque
import torch.optim as optim
import random
import matplotlib.pyplot as plt
#%matplotlib inline 

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

        # disturbance term
        self.d1 = 1*np.sin(self.X[2][0])**2 + 1*np.sin(self.X[2][0]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2,0])**2 + 1*np.sin(self.X[2,0]*16)
        self.d3 = 0.3
        
        extra_f = np.array([self.d1*np.cos(self.X[2][0]),self.d2*np.sin(self.X[2][0]),self.d3]).reshape(-1,1)
        extra_g = np.array([ [0,0],[0,0],[0,0] ])
        
        self.X = self.X + (extra_f + extra_g @ U + self.f + self.g @ U)*self.dt
    
        self.X[2] = wrap_angle(self.X[2])
        
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
        GP_list.append(gp)
    return GP_list

@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ConvergenceWarning)
def update_GP_dynamics(GP_list,X,Y, index, curr_X):
    # X, Y = filter_data(X,Y,curr_X)
    GP_list[index].fit(np.asarray(X).reshape(-1,1),np.asarray(Y).reshape(-1,1))


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
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def test_GP_dynamics(GP_list,X,index):
    Y = []
    Y_std = []
    for i in X:
        y_pred, sigma_pred = GP_list[index].predict(np.array([i]).reshape(-1, 1),return_std=True)
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)


def compute_CLF_GP(X,Xd):
    V = (np.linalg.norm(X-Xd))*2
    return V

def CBF_loss(agent,target,u,w):

    alpha = 0.1 #1.0 does not work

    h1 , h1A , h1B = CBF1_loss(agent,target)
    h2 , h2A , h2B = CBF2_loss(agent,target)
    h3 , h3A , h3B = CBF3_loss(agent,target)

    U = np.array([u,w]).reshape(-1,1)
    h1_ = h1A @ U + h1B + alpha * h1
    h2_ = h2A @ U + h2B + alpha * h2
    h3_ = h3A @ U + h3B + alpha * h3

    if h1_ > 0 and h2_ > 0 and h3_ > 0:
        return True, h1_, h2_, h3_
    else:
        return False, h1_, h2_, h3_

def CBF1_loss(agent,target):
    h = max_D**2 - np.linalg.norm( agent.X[0:2,0] - target.X[:,0] )**2
    h_dot_B = 2*( agent.X[0:2,0] - target.X[:,0] ) @ target.V  - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f[0:2,:]  ) - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f_corrected[0:2,:]  )
    h_dot_A = - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g[0:2,:]  ) - 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g_corrected[0:2,:]  )
    
    return h, h_dot_A, h_dot_B

def CBF2_loss(agent,target):
    h = np.linalg.norm( agent.X[0:2,0] - target.X[:,0] )**2 - min_D**2

    h_dot_B = - 2*( agent.X[0:2,0] - target.X[:,0] ) @ target.V  + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f[0:2,:]  ) + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.f_corrected[0:2,:]  )
    h_dot_A =   2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g[0:2,:]  ) + 2*( agent.X[0:2,0] - target.X[:,0] ) @ ( agent.g_corrected[0:2,:]  )
    return h, h_dot_A, h_dot_B


def CBF3_loss(agent,target):
    theta = agent.X[2,0]
    # Direction Vector
    bearing_vector = np.array([np.cos(theta),np.sin(theta)]) @ (target.X[:,0] - agent.X[0:2,0])/np.linalg.norm(target.X[:,0] - agent.X[0:2,0])
    h = bearing_vector - np.cos(FoV/2)

    dir_vector = np.array([np.cos(theta),np.sin(theta)])
    
    p_transpose = np.array([target.X[0,0],target.X[1,0]]) - np.array([agent.X[0,0],agent.X[1,0]])
    p = p_transpose.reshape(-1,1)
    factor = dir_vector/np.linalg.norm(agent.X[0:2]-target.X[:,0]) - ( (dir_vector @ p) * p_transpose )/2/np.linalg.norm(agent.X[0:2,0]-target.X[:,0])/(np.linalg.norm(agent.X[0:2,0]-target.X[:,0])**2)

    factor_2 = ( (target.X[:,0] - agent.X[0:2,0])/np.linalg.norm(target.X[:,0] - agent.X[0:2,0]) @ np.array([ [-np.sin(theta)],[np.cos(theta)] ]) )
    h_dot_B = factor @ target.V.reshape(-1,1) - factor @ agent.f[0:2,:] - factor @ agent.f_corrected[0:2,:] + factor_2 * agent.f[2,:] + factor_2 * agent.f_corrected[2,:]
    h_dot_A = -factor @ agent.g[0:2,:] - factor @ agent.g_corrected[0:2,:] + factor_2 * agent.g[2,:] + factor_2 * agent.g_corrected[2,:]
    return h, h_dot_A, h_dot_B

def CLF_loss(agent,target):
    
    avg_D = (min_D + max_D)/2.0
    V = (np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) - avg_D)**2

    # V_dot = A*U + B
    factor = (np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) - avg_D)/np.linalg.norm( agent.X[0:2,0] - target.X[:,0] ) * 2 * ( np.array([ agent.X[0,0], agent.X[1,0] ]) - np.array([target.X[0,0],target.X[1,0]]) )
    V_dot_B = factor @ agent.f[0:2,:] + factor @ agent.f_corrected[0:2,:]
    V_dot_A = factor @ agent.g[0:2,:] + factor @ agent.g_corrected[0:2,:]

    return V, V_dot_A, V_dot_B


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
    return v, omega #np.array([v,omega])

def CBF_CLF_QP(agent,target):

    u, w = nominal_controller(agent,target)
    U = np.array([u,w]).reshape(-1,1)

    alpha = 0.1 #1.0 does not work
    k = 0.1

    h1,h1A,h1B = CBF1_loss(agent,target)
    h2,h2A,h2B = CBF2_loss(agent,target)
    h3,h3A,h3B = CBF3_loss(agent,target)

    V,VA,VB = CLF_loss(agent,target)

    G = np.array([ [ VA[0] , VA[1], -1.0   ],
                   [ -h1A[0], -h1A[1], 0.0 ] ,
                   [ -h2A[0], -h2A[1], 0.0 ] ,
                   [ -h3A[0], -h3A[1], 0.0 ]    
                    ])

    h = np.array([-VB - k*V ,
                 h1B + alpha*h1  ,
                 h2B + alpha*h2  ,
                 h3B + alpha*h3 ]).reshape(-1,1)

    #Convert numpy arrays to cvx matrices to set up QP
    G = matrix(G,tc='d')
    h = matrix(h,tc='d')

    # Cost matrices
    P = matrix(np.diag([2., 2., 6.]), tc='d')
    q = matrix(np.array([ -2*u, -2*w, 0 ]))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u_bar = sol['x']

    if sol['status'] != 'optimal':
        return False, 0, 0

    return True, u_bar[0], u_bar[1]



def compute_CBF(angle_diff,distance):

    FoV = 30*np.pi/180
    max_D = 3
    min_D = 0.7

    if np.abs(angle_diff)>FoV:
        barrier_angle = -1 #-np.abs(FoV-np.abs(angle_diff))/FoV  #   -1
    else:
        barrier_angle = np.abs(FoV-np.abs(angle_diff))/FoV
    
    mean_D = (max_D-min_D)/2
    c = mean_D**2
    mean_dist = (min_D+max_D)/2

    if distance>max_D:
        barrier_distance = -1
    elif distance<min_D:
        barrier_distance = -1
    else:
        barrier_distance = c - (distance - mean_dist)**2
    # barrier_distance = c - (distance - mean_dist)**2  # continuous variation
        
    barrier = np.abs(barrier_angle)*np.abs(barrier_distance)
    if (barrier_angle<0) or (barrier_distance<0):
        barrier = -barrier

    return barrier

def compute_CLF(angle_diff,distance):

    max_D = 3
    min_D = 0.7

    mean_D = (max_D+min_D)/2

    w_angle = 1.0
    w_distance = 1.0
    V = w_angle*angle_diff**2 + w_distance*(distance - mean_D)**2

    return V


def compute_reward_nominal(angle_diff,distance):

    FoV = 30*np.pi/180
    max_D = 3
    min_D = 0.7

    if np.abs(angle_diff)>FoV:
        barrier_angle = -1 #-np.abs(FoV-np.abs(angle_diff))/FoV  #   -1
    else:
        barrier_angle = np.abs(FoV-np.abs(angle_diff))/FoV
    
    mean_D = (max_D-min_D)/2
    c = mean_D**2
    mean_dist = (min_D+max_D)/2

    if distance>max_D:
        barrier_distance = -1
    elif distance<min_D:
        barrier_distance = -1
    else:
        barrier_distance = c - (distance - mean_dist)**2
    # barrier_distance = c - (distance - mean_dist)**2  # continuous variation
        
    barrier = np.abs(barrier_angle)*np.abs(barrier_distance)
    if (barrier_angle<0) or (barrier_distance<0):
        barrier = -barrier

    return barrier
    
def compute_reward(F_X,T_X,F_X_prev,T_X_prev,type='nominal',debug=False):

    alpha_cbf = 0.4#3.0#0.4
    alpha_clf = 0.4
    
    #h2 - current time
    beta = np.arctan2(T_X[1,0]-F_X[1,0],T_X[0,0]-F_X[0,0])
    angle_diff = wrap_angle(beta - F_X[2,0])
    distance = np.sqrt( (T_X[0,0]-F_X[0,0])**2 + (T_X[1,0]-F_X[1,0])**2 )

    h2 = compute_CBF(angle_diff,distance)
    V2 = compute_CLF(angle_diff,distance)
    reward_nominal_2 = compute_reward_nominal(angle_diff,distance)  # same 

    #h1 - prev time
    beta = np.arctan2(T_X_prev[1,0]-F_X_prev[1,0],T_X_prev[0,0]-F_X_prev[0,0])    
    angle_diff = wrap_angle(beta - F_X_prev[2,0])    
    distance = np.sqrt( (T_X_prev[0,0]-F_X_prev[0,0])**2 + (T_X_prev[1,0]-F_X_prev[1,0])**2 )
    
    h1 = compute_CBF(angle_diff,distance)
    V1 = compute_CLF(angle_diff,distance)
    reward_nominal_1 = compute_reward_nominal(angle_diff,distance)

    reward_CBF = h2 - h1 + alpha_cbf*h1    
    reward_CLF = -(V2 - V1) - alpha_clf*V2

    if debug:
        print(f"h:{h1}, h2:{h2}, CBF:{reward_CBF}")



    if type=='nominal':
        return reward_nominal_2
    elif type=='CBF':
        # print(f"nominal:{reward_nominal_2}, CBF:{reward_CBF}")
        return reward_CBF
    elif type=='CLF':
        return reward_CLF
    else:
        print('undefined reward type')
        exit()
    



#Construct Neural Networks

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc_units=256, fc1_units=256):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x)) * self.max_action

# Q1-Q2-Critic Neural Network

class Critic_Q(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        super(Critic_Q, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc1_units)
        self.l5 = nn.Linear(fc1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2
upper_bound = 3
lower_bound = -3

dt = 0.1

class TD3:
    def __init__(
        self,name,env,
        load = False,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4,
        lr_critic = 3e-4,
        batch_size = 100,
        buffer_capacity = 1000000,
        tau = 0.02,  #soft update parameter
        random_seed = 666,
        cuda = True,
        policy_noise=0.2, 
        std_noise = 0.1,
        noise_clip=0.5,
        policy_freq=2, #target network update period
        plot_freq = 1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.env = env
        self.upper_bound = float(3)#float(self.env.action_space.high[0]) #action space upper bound
        self.lower_bound = float(-3)#float(self.env.action_space.low[0])  #action space lower bound
        self.create_actor()
        self.create_actor_temp()
        self.create_critic()
        self.lr_actor = lr_actor
        self.lr_actor_step = lr_actor/10
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen = buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.name = name
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise   
        self.plot_freq = plot_freq
        
    def create_actor(self):
        params = {
            'state_size':      5,
            'action_size':     2,
            'max_action':      self.upper_bound
        }
        self.actor = Actor(**params).to(self.device)
        self.actor_target = Actor(**params).to(self.device)

    def create_actor_temp(self):
        params = {
            'state_size':      5,
            'action_size':     2,
            'max_action':      self.upper_bound
        }
        self.actor_temp = Actor(**params).to(self.device)
        self.actor_target_temp = Actor(**params).to(self.device)

    def create_critic(self):
        params = {
            'state_size':      5,
            'action_size':     2
        }
        self.critic = Critic_Q(**params).to(self.device)
        self.critic_target = Critic_Q(**params).to(self.device)

    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def add_to_replay_memory(self, state, action, reward, next_state, done):
      #add samples to replay memory
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self,timesteps=None):
      #random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)

        # random_sample = self.replay_memory_buffer

        return random_sample

    def learn_and_update_weights_by_replay(self,training_iterations):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        iter_count = 1

        changed = False

        if len(self.replay_memory_buffer) < self.batch_size:
            return
        for it in range(training_iterations):
            if iter_count>200:
                break;
            iter_count += 1
            # print("iter count",iter_count)
            #NEW
            lr_actor = self.lr_actor

            mini_batch = self.get_random_sample_from_replay_mem()
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)

            # Training and updating Actor & Critic networks.
            #Train Critic
            target_actions = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise).to(self.device)

            #clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            #Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            #Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            #Train Actor
            # Delayed policy updates

            if it % self.policy_freq == 0:
                # Minimize the loss
                actions = self.actor(state_batch)
                actor_loss,_ = self.critic(state_batch, actions)
                actor_loss = - actor_loss.mean()

                improved = False
                max_iter = 100
                test_reward_prev = eval_policy_sim(self, seed=88, eval_episodes=1)
                test_reward_prev_first = test_reward_prev
                # print("here 1")
                
                # copy from actor to actor_temp
                for target_param, local_param in zip(self.actor_temp.parameters(), self.actor.parameters()):
                    target_param.data.copy_(local_param.data)

                while not improved and lr_actor>0.00003:
                    lr_actor = lr_actor - self.lr_actor_step
                    
                    for g in self.act_opt.param_groups:                        
                        g['lr'] = lr_actor

                    # Optimize the actor               
                    self.act_opt.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.act_opt.step()

                    test_reward_new = eval_policy_sim(self, seed=88, eval_episodes=1)
                    # print(f"In TRAIN:{lr_actor}, test_reward_new:{test_reward_new}, test_reward_prev:{test_reward_prev}")
                    if (test_reward_new > test_reward_prev):
                        improved = True
                        changed = True
                        break

                    # copy from actor_temp to actor
                    for target_param, local_param in zip(self.actor.parameters(), self.actor_temp.parameters()):
                        target_param.data.copy_(local_param.data)

                test_reward_new = eval_policy_sim(self, seed=88, eval_episodes=1)
                
                if test_reward_new<test_reward_prev_first-0.02:
                    print("reward before",test_reward_prev_first)
                    print("reward after",test_reward_new)
                    exit()


                # # Optimize the actor               
                # self.act_opt.zero_grad()
                # actor_loss.backward()
                # self.act_opt.step()

                #Soft update target models
                self.soft_update_target(self.critic, self.critic_target)
                self.soft_update_target(self.actor, self.actor_target)

        if changed==True:
            print("policy changed")

    def soft_update_target(self,local_model,target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def policy(self,state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return np.squeeze(actions)



def eval_policy(agent, seed, eval_episodes=10):
    # eval_env = eval_env
    avg_reward = 0.

    plt.ion()

    fig = plt.figure()
    ax = plt.axes(xlim=(0,20),ylim=(-10,10))

    lines, = ax.plot([],[],'o-')
    areas, = ax.fill([],[],'r',alpha=0.1)
    bodyF = ax.scatter([],[],c='r',s=10)
    
    bodyT = ax.scatter([],[],c='g',s=10)


    for _ in range(eval_episodes):
        # state, done = eval_env.reset(), False
        agentF = follower(np.array([0,0.2,0]),dt)
        agentT = target(np.array([1,0]),dt)
        done = False

        t = 0

        T_ns_prev = agentT.X
        F_ns_prev = agentF.X

        T = agentT.X
        F = agentF.X    
        state = np.array([F[0],F[1],F[2],T[0],T[1]])

        while not done:
            action = agent.policy(np.array(state))

            uL = 0.1
            vL = 0.2*np.sin(np.pi*t/5) #  0.1

            T = agentT.X
            F = agentF.X    
            # print(T)
            #print(F)
            state = np.array([F[0],F[1],F[2],T[0],T[1]])
            # state, reward, done, _ = eval_env.step(action)

            u = action[0]
            v = action[1]
            
            T_ns = agentT.step(0.1,0.1)        
            F_ns = agentF.step(u,v)
            reward = compute_reward(F_ns,T_ns,F_ns_prev,T_ns_prev)
            
            #print(reward)
            
            lines, areas, bodyF = agentF.render(lines,areas,bodyF)
            bodyT = agentT.render(bodyT)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if reward<0:
                done = True

            avg_reward += reward

            T_ns_prev = T_ns
            F_ns_prev = F_ns

            t += dt

    avg_reward /= eval_episodes

    plt.close()

    print("---------------------------------------")

    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(args):
    env = gym.make('BipedalWalker-v3')
    agent = TD3(args.rl_name, env, gamma=args.gamma, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                batch_size = args.batch_size, buffer_capacity=args.buffer_capacity, tau=args.tau, random_seed=args.seed,
                policy_freq=args.policy_freq, plot_freq=args.plot_freq)
    time_start = time.time()        # Init start time
    ep_reward_list = []
    avg_reward_list = []
    total_timesteps = 0

    epsilon = 0.1
    epsilon2 = 0.4
    epsilon2_decay = 0.999
    epsilon_decay = 0.999

    epsilon_min = 0.05
    epsilon2_min = 0.1#0.05

    timestep_list = []
    avg_timestep_list = []

    reward_list = []

    N = 3

    GP_list = build_GP_model(N)

    for ep in range(args.total_episodes):

        reward_episode = []
        if ep % args.plot_freq ==0:
            plt.ion()

            fig = plt.figure()
            ax = plt.axes(xlim=(0,20),ylim=(-10,10))

            lines, = ax.plot([],[],'o-')
            areas, = ax.fill([],[],'r',alpha=0.1)
            bodyF = ax.scatter([],[],c='r',s=10)
            
            bodyT = ax.scatter([],[],c='g',s=10)


        # state = env.reset()
        episodic_reward = 0
        timestep = 0

        agentF = follower(np.array([0,0.2,0]),dt)
        agentT = target(np.array([1,0]),dt)

        T_ns_prev = agentT.X
        F_ns_prev = agentF.X

        T = agentT.X
        F = agentF.X    
        state = np.array([F[0,0],F[1,0],F[2,0],T[0,0],T[1,0]])

        t = 0

        uL = 0.1
        vL = 0.2*np.sin(np.pi*t/5) #  0.1

        # epsilon = epsilon*epsilon_decay
        # if epsilon<epsilon_min:
        #     epsilon = epsilon_min
        
        # epsilon2 = epsilon2*epsilon2_decay
        # if epsilon2<epsilon2_min:
        #     epsilon2 = epsilon2_min

        epsilon = 0.1
        epsilon2 = 0.4
        epsilon2_decay = 0.999
        epsilon_decay = 0.999

        epsilon_min = 0.05
        epsilon2_min = 0.1#0.05

        obs_d1 = []
        obs_d2 = []
        obs_d3 = []

        true_d1 = []
        true_d2 = []
        true_d3 = []

        pred_d1 = []
        pred_d2 = []
        pred_d3 = []

        debug_value = []

        Xgp = []

        for st in range(args.max_steps):

            epsilon = epsilon*epsilon_decay
            if epsilon<epsilon_min:
                epsilon = epsilon_min
            
            epsilon2 = epsilon2*epsilon2_decay
            if epsilon2<epsilon2_min:
                epsilon2 = epsilon2_min

            uL = 0.1
            vL = 0.2*np.sin(np.pi*t/5) #  0.1

            if t==0:
                d1 = 0
                d2 = 0
                d3 = 0
            else:
                d, d_std = predict_GP_dynamics( GP_list,N,agentF.X[2,0]  )
                d1 = d[0]
                d2 = d[1]
                d3 = d[2]       

            pred_d1.append(d1)
            pred_d2.append(d2)
            pred_d3.append(d3)

            #exact
            # agentF.f_corrected = np.array([agentF.d1*np.cos(agentF.X[2][0]),agentF.d2*np.sin(agentF.X[2][0]),agentF.d3]).reshape(-1,1)
            # predicted
            agentF.f_corrected = np.array([d1*np.cos(agentF.X[2][0]),d2*np.sin(agentF.X[2][0]),d3]).reshape(-1,1)
            # ignored
            # agentF.f_corrected = np.array([0*np.cos(agentF.X[2][0]),0*np.sin(agentF.X[2][0]),0]).reshape(-1,1)
            agentF.g_corrected = np.array([ [0,0],[0,0],[0,0] ])

            Xgp.append(agentF.X[2,0])

            # Select action randomly or according to policy
            # if total_timesteps < args.start_timestep:
            rand_sample = np.random.random()
            # if (np.random.random()<epsilon):
            if rand_sample<epsilon:
                found_valid = False
                max_iter = 10
                count_iter = 0
                prediction_boundary_prev = 100
                # while not found and count_iter<max_iter:
                while count_iter<max_iter:
                    count_iter += 1
                    u = -3 + 6*random.uniform(0,1)
                    v = -3 + 6*random.uniform(0,1)
                    action_temp = [u,v]
                    valid_input, h1_, h2_, h3_ = CBF_loss(agentF,agentT,u,v)
                    prediction_boundary = min(h1_,h2_,h3_)
                    # print("PREDICTION",reward_prediction_prev)
                    if valid_input>=0:
                        found_valid = True
                        if prediction_boundary<prediction_boundary_prev:  # take the one nearest the boundary
                            prediction_boundary_prev = prediction_boundary
                            action = action_temp
                # print("PREDICTION",reward_prediction_prev)
                # print(f"action: u:{action[0]}, v:{action[1]}")
                if not found_valid: # can't help it
                    print("********************  COULD NOT FIND valid boundary random sample in 100 trials ************************")
                    action = agent.policy(state)
                else:
                    print("FOUND VALID BOUNDARY")
                

                # u = -3 + 6*random.uniform(0,1)
                # v = -3 + 6*random.uniform(0,1)
                # action = [u,v]
                #Choose epsilon based on CBF boundary

            elif rand_sample < epsilon2:
                reward_prediction = -1
                d_std = 1.0
                valid_input = False
                max_iter = 10
                count_iter = 0
                while not valid_input<0 and d_std >0.4 and count_iter<max_iter:
                    count_iter += 1
                    u = -3 + 6*random.uniform(0,1)
                    v = -3 + 6*random.uniform(0,1)
                    action_temp = [u,v]
                    valid_input, h1_, h2_, h3_ = CBF_loss(agentF,agentT,u,v)
                if not valid_input:
                    print("********************  COULD NOT FIND valid random sample in 100 trials ************************")
                    action = agent.policy(state)
                else:
                    print("FOUND VALID")
                    action = action_temp
            else:   
                action = agent.policy(state)

            # still need.. adversarial actions.. explorartion not clear then.. how to do it??
            # have F_ns_prev, T_ns_prev: oh just do agent .step and do it?? assumes known dynamics then?? 
            # But do noit know uL, vL?? Now what???

            # action = agent.policy(state)

            # Recieve state and reward from environment.
            # next_state, reward, done, info = env.step(action)

            FX_prev = agentF.X


            u = action[0]
            v = action[1]
            T_ns = agentT.step(uL,vL)  #agentT.step(0.2,0.5)        
            U = np.array([u,v]).reshape(-1,1)
            F_ns = agentF.step(U)

            FX = agentF.X
            # print("d1 from F",agentF.d1)
            if (np.abs(np.cos(FX_prev[2,0]))>0.0001):
                d1_obs = ( (FX[0,0]-FX_prev[0,0])/dt - u*np.cos(FX_prev[2,0]) )/np.cos(FX_prev[2,0])
                # print("d1_obs",d1_obs)
                # print(f"d1_obs: {d1_obs}, X:{FX_prev}, X_next:{FX}, gu :{u*np.cos(FX_prev[2])}, f:{ (FX[0]-FX_prev[0])/dt - u*np.cos(FX_prev[2]) } ")
            else:
                # print("0000")
                d1_obs = 0
            if (np.abs(np.sin(FX_prev[2,0]))>0.0001):
                d2_obs = ( (FX[1,0]-FX_prev[1,0])/dt - u*np.sin(FX_prev[2,0]) )/np.sin(FX_prev[2,0])
            else:
                d2_obs = 0
            d3_obs =  wrap_angle(FX[2,0]-FX_prev[2,0])/dt - v
            # print("d1_obs",d1_obs)

            obs_d1.append(d1_obs)
            obs_d2.append(d2_obs)
            obs_d3.append(d3_obs)

            debug_value.append(FX_prev[2,0])    

            # update_GP_dynamics(GP_list,N,np.array([FX_prev[2],FX_prev[2],FX_prev[2]] ), np.array([d1_obs,d2_obs,d3_obs]))
            update_GP_dynamics(GP_list,Xgp, obs_d1, 0, agentF.X[2,0])
            update_GP_dynamics(GP_list,Xgp, obs_d2, 1, agentF.X[2,0])
            update_GP_dynamics(GP_list,Xgp, obs_d3, 2, agentF.X[2,0])

            reward = compute_reward(F_ns,T_ns,F_ns_prev,T_ns_prev)

            reward_episode.append(reward)

            reward_prev = reward
            next_state = np.array([FX[0,0],FX[1,0],FX[2,0],agentT.X[0,0],agentT.X[1,0]])

            if ep % args.plot_freq ==0:
                lines, areas, bodyF = agentF.render(lines,areas,bodyF)
                bodyT = agentT.render(bodyT)
                
                fig.canvas.draw()
                fig.canvas.flush_events()


            if reward<0:
                done = True
            else:
                done = False
            # done = False
            # print("target state",T_ns)

            #change original reward from -100 to -5 and 5*reward for other values
            episodic_reward += reward
            # if reward == -100:
            #     reward = -5
            # else:
            #     reward = 5 * reward
            agent.add_to_replay_memory(state, action, reward, next_state, done)          
            # End this episode when `done` is True
            if done:
                break
            state = next_state
            timestep += 1     
            total_timesteps += 1

            T_ns_prev = T_ns
            F_ns_prev = F_ns

            t += dt

        if ep % args.plot_freq ==0:
            plt.close()

        reward_list.append(reward_episode)

        ep_reward_list.append(episodic_reward)
        # Mean of last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)

        timestep_list.append(timestep)

        avg_timestep = np.mean(timestep_list[-100:])
        avg_timestep_list.append(avg_timestep)

        # if avg_reward > 290:
        # if (ep % 1 == 0 and ep>1):# or episodic_reward>70 :
        if episodic_reward>1000:
            # print("here 2")
            #test_reward = eval_policy(agent, seed=88, eval_episodes=1)
            #if test_reward > 30000:
                # print("here 3")
                #final_test_reward = eval_policy(agent, seed=88, eval_episodes=10)
                #if final_test_reward > 300:
                    print("===========================")
                    print('Task Solved')
                    print("===========================")
                    #save weights
                    torch.save(agent.actor.state_dict(), 'actor_CBF.pth')
                    break     
            # if episodic_reward>100:   
            #     torch.save(agent.actor.state_dict(), 'actor_CBF.pth')    
        s = (int)(time.time() - time_start)
        agent.learn_and_update_weights_by_replay(timestep)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Episode Reward: {:.2f}, Moving Avg.Reward: {:.2f}, Fail Time: {:.2f} , epsilon: {:.2f}, Time: {:02}:{:02}:{:02}'
                .format(ep, total_timesteps, timestep,
                      episodic_reward, avg_reward, avg_timestep, epsilon, s//3600, s%3600//60, s%60)) 

        plt.figure()
        plt.ioff()
        plt.subplot(3,1,1)
        plt.plot(obs_d1,'g*')
        # plt.plot(true_d1,'r')
        plt.plot(pred_d1,'c')

        plt.subplot(3,1,2)
        plt.plot(obs_d2,'g*')
        # plt.plot(true_d2,'r')
        plt.plot(pred_d2,'c')

        plt.subplot(3,1,3)
        plt.plot(obs_d3,'g*')
        # plt.plot(true_d3,'r')
        plt.plot(pred_d3,'c')
        plt.show()
        # plt.pause(1.0)
        # plt.close()

        plt.ion()
        # print("Continue?")
        # input_key = input()
        # print(input_key)
        # if input_key == 's':
        #     break
    # Plotting graph
    # Episodes versus Avg. Rewards

    


    plt.ioff()
    print("DONE!!")
    plt.figure()

    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Epsiodic Reward")
    plt.show()

    print("DONE!!")
    plt.figure()

    plt.plot(timestep_list)
    plt.xlabel("Episode")
    plt.ylabel("Time to Failure")
    plt.show()

    print("DONE!!")

    plt.figure()
    for reward_episode in reward_list:
        plt.plot(reward_episode)
    plt.show()

    print("reward list",reward_list)

    f = open('data_new.dat','wb')

    for reward_episode in reward_list:
        np.savetxt(f,np.asarray(reward_episode))
        f.write('\n \n \n')
    f.close()


    env.close()

import argparse
parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="BipedalWalkerHardcore-v3")
parser.add_argument('--rl-name', default="td3")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',help='target smoothing coefficient(τ)')
parser.add_argument('--lr-actor', type=float, default=0.0003, metavar='G',help='learning rate of actor')
parser.add_argument('--lr-critic', type=float, default=0.0003, metavar='G',help='learning rate of critic')
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--seed', type=int, default=123456, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size (default: 256)')
parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='N', help='buffer_capacity')
parser.add_argument('--max-steps', type=int, default=10000, metavar='N',help='maximum number of steps of each episode')
parser.add_argument('--total-episodes', type=int, default=1000, metavar='N',help='total training episodes')
parser.add_argument('--policy-freq', type=int, default=2, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
args = parser.parse_args("")

train(args)

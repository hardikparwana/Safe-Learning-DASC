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



def true_d(theta,index):
    if index==0:
        return 1*np.sin(theta)**2 + 1*np.sin(theta*16)
    elif index==1:
        return 1*np.sin(theta)**2 + 1*np.sin(theta*16)
    else:
        return theta*0+0.3

class DoubleUnicycle2D:
    
    def __init__(self,X0,dt):
        self.X  = X0
        self.dt = dt
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        
    def step(self,a,w):
        
        gu = np.array([self.X[3]*np.cos(self.X[2]),self.X[3]*np.sin(self.X[2]),w,a])
        
        # disturbance term
        self.d1 = 1*np.sin(self.X[2])**2 + 1*np.sin(self.X[2]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2])**2 + 1*np.sin(self.X[2]*16)
        self.d3 = 0.3
        
        f = np.array([self.d1*np.cos(self.X[2]),self.d2*np.sin(self.X[2]),self.d3])
        # print(f"true d1:{self.d1} X: {self.X} , f:{d1*np.cos(self.X[2])}, g:{u*np.cos(self.X[2])}, cos term: {np.cos(self.X[2])} ")
        self.X = self.X + (f + gu)*self.dt

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
        FoV = np.pi/3

        x = np.array([self.X[0],self.X[1]])
        theta = self.X[2]
        theta1 = theta + FoV/2
        theta2 = theta - FoV/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + length*e1
        P2 = x + length*e2  

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        triangle_v = [ x,P1,P2,x ]  

        lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)

        length2 = 3

        # scatter plot update
        body.set_offsets([x[0],x[1]])
#         sc.set_offsets(np.c_[x,y])

        return lines, areas, body


class follower:
    
    def __init__(self,X0,dt):
        self.X  = X0
        self.dt = dt
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        
    def step(self,u,w):
        
        gu = np.array([u*np.cos(self.X[2]),u*np.sin(self.X[2]),w])
        
        # disturbance term
        self.d1 = 1*np.sin(self.X[2])**2 + 1*np.sin(self.X[2]*16) #sin(u)
        self.d2 = 1*np.sin(self.X[2])**2 + 1*np.sin(self.X[2]*16)
        self.d3 = 0.3
        
        f = np.array([self.d1*np.cos(self.X[2]),self.d2*np.sin(self.X[2]),self.d3])
        # print(f"true d1:{self.d1} X: {self.X} , f:{d1*np.cos(self.X[2])}, g:{u*np.cos(self.X[2])}, cos term: {np.cos(self.X[2])} ")
        self.X = self.X + (f + gu)*self.dt

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
        FoV = np.pi/3

        x = np.array([self.X[0],self.X[1]])
        theta = self.X[2]
        theta1 = theta + FoV/2
        theta2 = theta - FoV/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + length*e1
        P2 = x + length*e2  

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        triangle_v = [ x,P1,P2,x ]  

        lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)
        print("triangle_v",triangle_v)

        length2 = 3

        # scatter plot update
        body.set_offsets([x[0],x[1]])
#         sc.set_offsets(np.c_[x,y])

        return lines, areas, body
       
class target:
    
    def __init__(self,X0,dt):
        self.X = X0
        self.dt = dt
        self.t0 = 0
        self.speed = 0
        self.theta = 0
        
    def step(self,a,alpha): #Just holonomic X,T acceleration
        
        if (self.speed<2):
            self.speed = self.speed + a*self.dt
        
        self.X = self.X + np.array([a,alpha])*dt
        return self.X

    def step_sim(self,a,alpha): #Just holonomic X,T acceleration
        
        X_sim = self.X + np.array([a,alpha])*dt
        return X_sim
    
    def render(self,body):
        length = 3
        FoV = np.pi/3

        x = np.array([self.X[0],self.X[1]])

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
def update_GP_dynamics(GP_list,X,Y, index):
    # print("X",X)
    # print("Y",Y)
    # for i in range(N):
    #     GP_list[i].fit(np.array([X[i]]).reshape(-1, 1),np.array([Y[i]]).reshape(-1, 1))
    GP_list[index].fit(np.asarray(X).reshape(-1,1),np.asarray(Y).reshape(-1,1))

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

def solve_QP(X,Xd,d1,d2,d3):
    #simple controller for now

    #Define gamma for the Lyapunov function
    k_omega = 2.5
    k_v = 0.5

    theta_d = np.arctan2(Xd[1]-X[1],Xd[0]-X[0])
    error_theta = wrap_angle( theta_d - X[2] )
    omega = k_omega*error_theta - d2

    distance = max(np.linalg.norm( X[0:2]-Xd ) - 0.3,0)

    v = k_v*( distance )*np.cos( error_theta )**2 - (d1 + d2)/2
    # print(f"w:{omega}, v:{v}")
    return v, omega #np.array([v,omega])





dt = 0.1
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

debug_value = []

Xgp = []

for k in range(0,50):

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

    Xgp.append(agentF.X[2])

    # print(f"True values:{}, {}, {}    predicted values:{}, {}, {}")

    u,w = solve_QP(agentF.X,agentT.X[0:2],d1,d2,d3)

    # print("d1",d1)
    # print("d2",d2)

    FX_prev = agentF.X

    

    agentT.step(uL,vL)  #agentT.step(0.2,0.5)        
    agentF.step(u,w)

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
    update_GP_dynamics(GP_list,Xgp, obs_d1, 0)
    update_GP_dynamics(GP_list,Xgp, obs_d2, 1)
    update_GP_dynamics(GP_list,Xgp, obs_d3, 2)

    lines, areas, bodyF = agentF.render(lines,areas,bodyF)
    bodyT = agentT.render(bodyT)
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    t = t + dt



print(len(pred_d1))
print(len(true_d1))

print(pred_d1)
print(true_d1)
print("DONE!!")
plt.ioff()

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
plt.show()








    
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from matplotlib.colors import LogNorm
rng = np.random.RandomState(0)

def test_GP_dynamics(GP,X):
    Y = []
    Y_std = []
    for i in X:
        y_pred, sigma_pred = GP.predict(np.array([i]).reshape(-1, 1),return_std=True)
        # print("i",i)
        # print("ypred ",y_pred[0])
        # print("sigma ",sigma_pred)
        try: 
            Y.append(y_pred[0][0])
            Y_std.append(sigma_pred[0])
        except:
            Y.append(y_pred[0])
            Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def eval_GP(X,y_true,param0,param1,param2):
    Y = []
    Y_std = []
    kern_new = param0 * RBF(length_scale=param1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=param2, noise_level_bounds=(1e-10, 1e+1))
    noise = 0.01
    gp = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)
    gp.fit(X.reshape(-1,1),y_true.reshape(-1,1))
    for j, i in enumerate(X):
        y_pred, sigma_pred = gp.predict(np.array([i]).reshape(-1, 1),return_std=True)
        try: 
            Y.append((y_pred[0][0]-y_true[j])**2)
            Y_std.append(sigma_pred[0])
        except:
            Y.append(y_pred[0]-y_true[j])
            Y_std.append(sigma_pred[0])
    return np.mean(np.asarray(Y))


def gradient_update(gp,beta=0.01):
    likelihood, gradient = gp.log_marginal_likelihood(theta=gp.kernel_.theta,eval_gradient=True)
    # print(f"param:{np.exp(gp.kernel_.theta)}, gradient:{gradient}, params:{gp.get_params()}")
    param = np.exp(gp.kernel_.theta) + beta*gradient
    if param[2]<0:
        param[2]=0.01
    if param[1]<0:
        param[1]=0.01
    kern_new = param[0] * RBF(length_scale=param[1], length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=param[2], noise_level_bounds=(1e-10, 1e+1))
    noise = 0.01
    gp_new = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)
    # print(f"param:{np.exp(gp_new.kernel_.theta)}, gradient:{gradient}, params:{gp_new.get_params()}")

    return gp_new



## Sim
theta = np.linspace(-1,2,20)
noise = 0.01
kern = 5.0 * RBF(length_scale=7, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.6, noise_level_bounds=(1e-10, 1e+1))
#kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp_optimal = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
gp_fixed = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)
gp_step = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)

f_theta =  1*np.sin(theta)**2 + 1*np.sin(theta)

# Fit optimal
time_start = time.time() 
gp_optimal.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
print("Optimal time fit",(time.time() - time_start))  # 0.352
print("Optimal parameter: ", np.exp(gp_optimal.kernel_.theta))

# #fit fixed
# time_start = time.time() 
gp_fixed.fit(theta.reshape(-1,1),f_theta.reshape(-1,1)) 
# print("Fixed time fit",(time.time() - time_start))  # 0.0014
# print("Fixed: param: ",np.exp(gp_fixed.kernel_.theta))

# #One step gradient descent update
# time_start = time.time() 
# for i in range(200):
#     time_start = time.time() 
#     gp_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
#     gp_step = gradient_update(gp_step)
#     print("Step time fit",(time.time() - time_start))  # 0.004
# gp_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
# print("Step time fit",(time.time() - time_start))
# print("Step: param: ",np.exp(gp_step.kernel_.theta))





# ## Plot

# # Optimal
# plt.figure()
# d, d_std = test_GP_dynamics( gp_optimal,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # Fixed
# plt.figure()
# d, d_std = test_GP_dynamics( gp_fixed,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # Step
# plt.figure()
# d, d_std = test_GP_dynamics( gp_step,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# plt.show()

# # theta = np.linspace(0, 5, 100)
theta = rng.uniform(0, 5, 20)[:, np.newaxis]
y = 0.5 * np.sin(3 * theta[:, 0]) + rng.normal(0, 0.5, theta.shape[0])
# # gp_fixed.fit(theta.reshape(-1,1),y.reshape(-1,1))
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp_fixed = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(theta, y)

# plt.figure()
# # param0 = np.linspace(0.1,7,50)
# # param1 = np.linspace(0.1,100,100)
# # param2 = np.linspace(0.03,1,100)
param1 = np.logspace(-2, 3, 30)
param2 = np.logspace(-2, 0, 30)

# # Param0, Param1, Param2 = np.meshgrid(param0,param1,param2)
Param1, Param2 = np.meshgrid(param1,param2)

LML = [[gp_fixed.log_marginal_likelihood(np.log([0.36, Param1[i, j], Param2[i, j]]))
        for i in range(Param1.shape[0])] for j in range(Param2.shape[1])]
LML = np.array(LML).T
vmin, vmax = (-LML).min(), (-LML).max()
vmax = 50
print(f"vmin:{vmin}, vmax:{vmax}")
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

plt.contour(Param1, Param2, -LML,
            levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("Log-marginal-likelihood")
plt.tight_layout()
# plt.show()


plt.figure()

MSE = [ [eval_GP(theta, f_theta, 1.225, Param1[i,j], Param2[i,j]) for i in range(Param1.shape[0])  ] for j in range(Param2.shape[1]) ]
MSE = np.array(MSE).T
vmin, vmax = (MSE).min(), (MSE).max()
vmax = 50
# level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
plt.contour(Param1, Param2, MSE)#,
            # levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar()

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("MSE error")
plt.tight_layout()
plt.show()




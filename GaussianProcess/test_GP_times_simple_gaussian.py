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
from matplotlib import cm
rng = np.random.RandomState(0)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

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

def eval_GP(X,y_true,param0,param1):
    Y = []
    Y_std = []
    kern_new = param0 * RBF(length_scale=param1, length_scale_bounds=(1e-2, 1e3))# + WhiteKernel(noise_level=param2, noise_level_bounds=(1e-10, 1e+1))
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

def eval_GP_lml(X,y_true,param0,param1):
    kern_new = param0 * RBF(length_scale=param1, length_scale_bounds=(1e-2, 1e3))# + WhiteKernel(noise_level=param2, noise_level_bounds=(1e-10, 1e+1))
    noise = 0.01
    gp = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)
    gp.fit(X.reshape(-1,1),y_true.reshape(-1,1))

    Y = []
    Y_std = []
    Y_log = []

    for j, i in enumerate(X):
        y_pred, sigma_pred = gp.predict(np.array([i]).reshape(-1, 1),return_std=True)
        
        try: 
            Y.append((y_pred[0][0]-y_true[j])**2)
            Y_std.append(sigma_pred[0])
            log_prob = -np.log(sigma_pred[0]*np.sqrt(2)*np.pi) - (( (y_true[j]-y_pred[0][0])/sigma_pred[0] )**2)/2.0
            Y_log.append(log_prob)
        except:
            Y.append(y_pred[0]-y_true[j])
            Y_std.append(sigma_pred[0])
            log_prob = -np.log(sigma_pred[0]*np.sqrt(2)*np.pi) - (( (y_true[j]-y_pred[0])/sigma_pred[0] )**2)/2.0
            Y_log.append(log_prob)
    # print("sum log",np.sum(np.asarray(log_prob)))
    # exit()
    return np.sum(np.asarray(Y_log))


def lml_gradient_update(gp,beta=0.01):
    likelihood, gradient = gp.log_marginal_likelihood(theta=gp.kernel_.theta,eval_gradient=True)
    # print(f"param:{np.exp(gp.kernel_.theta)}, gradient:{gradient}, params:{gp.get_params()}")
    param = np.exp(gp.kernel_.theta) + beta*gradient
    if param[1]<0:
        param[1]=0.01
    kern_new = param[0] * RBF(length_scale=param[1], length_scale_bounds=(1e-2, 1e3)) #+ WhiteKernel(noise_level=param[2], noise_level_bounds=(1e-10, 1e+1))
    noise = 0.01
    gp_new = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)
    # print(f"param:{np.exp(gp_new.kernel_.theta)}, gradient:{gradient}, params:{gp_new.get_params()}")

    return gp_new

def mse_lml_gradient_update(gp,theta,y_true,beta=0.05):

    num_params = 2
    params_log = gp.kernel_.theta
    params = np.exp(gp.kernel_.theta)
    print("params_org",params)
    log_prob = eval_GP_lml(theta,y_true,params[0],params[1])
    

    gradients = np.array([])
    for index, value in enumerate(params):
        # print("index",index)
        # change in param
        delta_log = 0.1

        # print("params ",params_log)

        # left value
        params_temp = np.copy(params_log)
        params_temp[index] = params_temp[index] - delta_log
        delta = np.exp(params_log[index]) - np.exp(params_temp[index])

        params_temp = np.exp(params_temp)
        log_prob_left = eval_GP_lml(theta,y_true,params_temp[0],params_temp[1])
        gradient_left = (log_prob - log_prob_left)/delta
        # print(f"log c:{log_prob}, l:{log_prob_left}, delta:{delta}")

        # right value
        params_temp = np.copy(params_log)
        params_temp[index] = params_temp[index] + delta_log
        delta = np.exp(params_temp[index]) - np.exp(params_log[index])

        params_temp = np.exp(params_temp)        
        log_prob_right = eval_GP_lml(theta,y_true,params_temp[0],params_temp[1])
        
        # print("delta",delta)
        gradient_right = (log_prob_right - log_prob)/delta
        # print(f"log c:{log_prob}, r:{log_prob_right}, delta:{delta}: param:{params_log}")

        # gradient
        gradients = np.append( gradients, (gradient_right + gradient_left)/2.0 )
        # print(f"gradients: index:{index}, left:{gradient_left}, right:{gradient_right}, central:{(gradient_right + gradient_left)/2.0}, log:{log_prob}, log_left:{log_prob_left}, log_right:{log_prob_right}")
    
    params = params + beta*gradients
    print(f"gradients :{gradients}, change: {beta*gradients}, new_param:{params}")    
    
    # exit()
    if params[0]<0:
        params[0]=0.01
    if params[1]<0:
        params[1]=0.01
    kern_new = params[0] * RBF(length_scale=params[1], length_scale_bounds=(1e-2, 1e3)) 
    # print("kernel",kern_new)
    gp_new = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)
    # print(gp_new.get_params())
    gp_new.fit(theta.reshape(-1,1),y_true.reshape(-1,1))
    # print("same params",np.exp(gp_new.kernel_.theta))

    return gp_new

def mse_gradient_update(gp,theta,y_true,beta=1.0):

    num_params = 2
    params_log = gp.kernel_.theta
    params = np.exp(gp.kernel_.theta)
    # print("params_org",params)
    mean_error = eval_GP(theta,y_true,params[0],params[1])

    gradients = np.array([])
    for index, value in enumerate(params):
        # print("index",index)
        # change in param
        delta_log = 5

        # left value
        params_temp = np.copy(params_log)
        params_temp[index] = params_temp[index] - delta_log
        delta = np.exp(params_log[index]) - np.exp(params_temp[index])
        params_temp = np.exp(params_temp)
        mean_error_left = eval_GP(theta,y_true,params_temp[0],params_temp[1])        
        gradient_left = (mean_error - mean_error_left)/delta

        # right value
        params_temp = np.copy(params_log)
        params_temp[index] = params_temp[index] + delta_log
        delta = np.exp(params_temp[index]) - np.exp(params_log[index])
        params_temp = np.exp(params_temp)
        mean_error_right = eval_GP(theta,y_true,params_temp[0],params_temp[1])        
        gradient_right = (mean_error_right - mean_error)/delta

        # gradient
        gradients = np.append( gradients, (gradient_right + gradient_left)/2.0 )
        # print(f"gradients: left:{gradient_left}, right:{gradient_right}, central:{(gradient_right + gradient_left)/2.0}")

    params = params + beta*gradients
    print(f"gradients :{gradients}, change: {beta*gradients}, new_param:{params}")    

    if params[0]<0:
        params[0]=0.01
    if params[1]<0:
        params[1]=0.01
    kern_new = params[0] * RBF(length_scale=params[1], length_scale_bounds=(1e-2, 1e3)) 
    gp_new = GaussianProcessRegressor(kernel=kern_new, alpha = noise, optimizer=None)

    return gp_new





# ## Sim
# theta = np.linspace(-1,2,20)
# noise = 0.01
# kern = 5.0 * RBF(length_scale=7, length_scale_bounds=(1e-2, 1e3)) #+ WhiteKernel(noise_level=0.6, noise_level_bounds=(1e-10, 1e+1))
# #kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp_optimal = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
# gp_fixed = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)
# gp_lml_step = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)
# gp_mse_step = GaussianProcessRegressor(kernel=kern, alpha = noise, optimizer=None)


# f_theta =  1*np.sin(theta)**2 + 1*np.sin(theta)

# # Fit optimal
# time_start = time.time() 
# gp_optimal.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
# print("Optimal time fit",(time.time() - time_start))  # 0.352
# print("Optimal parameter: ", np.exp(gp_optimal.kernel_.theta))

# #fit fixed
# time_start = time.time() 
# gp_fixed.fit(theta.reshape(-1,1),f_theta.reshape(-1,1)) 
# print("Fixed time fit",(time.time() - time_start))  # 0.0014
# print("Fixed: param: ",np.exp(gp_fixed.kernel_.theta))

# #One step lml gradient descent update
# time_start = time.time() 
# for i in range(200):
#     gp_lml_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
#     gp_lml_step = lml_gradient_update(gp_lml_step)
# gp_lml_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
# print("Step LML time fit",(time.time() - time_start))
# print("Step LML: param: ",np.exp(gp_lml_step.kernel_.theta))

# #One step mse gradient descent update
# time_start = time.time() 
# for i in range(100):
#     # time_start = time.time() 
#     gp_mse_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
#     # print("Step MSE: param: ",np.exp(gp_mse_step.kernel_.theta))
#     # gp_mse_step = mse_lml_gradient_update(gp_mse_step,theta=theta, y_true=f_theta)
#     gp_mse_step = mse_gradient_update(gp_mse_step,theta=theta, y_true=f_theta)
#     # print("Step time fit",(time.time() - time_start))  # 0.004
# gp_mse_step.fit(theta.reshape(-1,1),f_theta.reshape(-1,1))
# print("Step MSE time fit",(time.time() - time_start))
# print("Step MSE: param: ",np.exp(gp_mse_step.kernel_.theta))



# # ## Plot

# # Optimal
# plt.figure()
# d, d_std = test_GP_dynamics( gp_optimal,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.title("Optimal fit")
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # Fixed
# plt.figure()
# d, d_std = test_GP_dynamics( gp_fixed,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.title("Fixed fit")
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # LML Step
# plt.figure()
# d, d_std = test_GP_dynamics( gp_lml_step,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.title("LML Step fit")
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # MSE Step
# plt.figure()
# d, d_std = test_GP_dynamics( gp_mse_step,theta  )
# plt.plot(theta,f_theta,'r',label='True Dyanmics')
# plt.plot(theta,d,label='Estimated Dynamics')
# plt.title("MSE Step fit")
# plt.legend()
# plt.fill_between(theta, d - d_std, d + d_std, alpha=0.2, color = 'm')

# # plt.show()


# # exit()
plt.figure()
# # theta = np.linspace(0, 5, 100)
theta = rng.uniform(0, 5, 20)[:, np.newaxis]
y = 0.5 * np.sin(3 * theta[:, 0]) + rng.normal(0, 0.5, theta.shape[0])
# y = 1*np.sin(theta[:,0])**2 + 1*np.sin(theta[:,0]*16) + + rng.normal(0, 0.5, theta.shape[0])
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) #\
   # + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp_fixed = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.1).fit(theta, y)

# plt.figure()
# # param0 = np.linspace(0.1,7,50)
# # param1 = np.linspace(0.1,100,100)
param0 = np.logspace(0.1, 3, 30)
param1 = np.logspace(-2, 3, 30)

# # Param0, Param1, Param2 = np.meshgrid(param0,param1,param2)
Param0, Param1 = np.meshgrid(param0,param1)

LML = [[gp_fixed.log_marginal_likelihood(np.log([Param0[i, j], Param1[i, j]]))
        for i in range(Param0.shape[0])] for j in range(Param1.shape[1])]
LML = np.array(LML).T
print(LML.shape)
vmin, vmax = (-LML).min(), (-LML).max()
vmax = 50
print(f"vmin:{vmin}, vmax:{vmax}")
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

plt.contour(Param0, Param1, -LML,
            levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))

# z = gp_fixed.log_marginal_likelihood(np.log(Param0),np.log(Param1))
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("Log-marginal-likelihood")
plt.tight_layout()
# plt.show()


plt.figure()
# LML = LML.T
plt.imshow(-LML,extent=(np.amin(param0), np.amax(param0), np.amin(param1), np.amax(param1)), aspect = 'auto')#,extent=[pow(10,0.1), pow(10,3), pow(10,-2), pow(10,3)])
# plt.imshow(-LML,cmap=cm.RdBu)
plt.colorbar()
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("Log-marginal-likelihood Self")
plt.tight_layout()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Param0, Param1, -LML, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
# plt.xscale("log")
# ax.yaxis.set_scale('log')


# plt.figure()

# MSE = [ [eval_GP_lml(theta, f_theta, Param0[i,j], Param1[i,j]) for i in range(Param0.shape[0])  ] for j in range(Param1.shape[1]) ]
# MSE = np.array(MSE).T
# vmin, vmax = (MSE).min(), (MSE).max()
# vmax = 50
# # level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
# plt.contour(Param0, Param1, MSE)#,
#             # levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
# plt.colorbar()

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Length-scale")
# plt.ylabel("Noise-level")
# plt.title("MSE error")
# plt.tight_layout()
plt.show()




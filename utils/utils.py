from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import math

def wrap_angle(angle):
    if angle>np.pi:
        angle = angle - 2*np.pi
    if angle<-np.pi:
        angle = angle + 2*np.pi
    return angle

def nCr(n,r):
    return math.factorial(n)*math.factorial()


def build_GP_model(N):
    GP_list = []
    noise = 0.01
    for i in range(N):
        kern = 1.0 * RBF(length_scale=2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
        gp = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=1)
        GP_list.append(gp)
    return GP_list

def filter_data(obs_x,obs_value,curr_x,thres=0.4):
    filter_x = []
    filter_value = []
    for index, t in enumerate(obs_x):
        if np.abs(t-curr_x)<thres:
            filter_x.append(t)
            filter_value.append(obs_value[index])
    return filter_x, filter_value

@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ConvergenceWarning)
def update_GP_dynamics(GP_list,X,Y, index, curr_X):
    X, Y = filter_data(X,Y,curr_X)
    GP_list[index].fit(np.asarray(X).reshape(-1,1),np.asarray(Y).reshape(-1,1))

def predict_GP_dynamics(GP_list,N,X):
    Y = []
    Y_std = []
    for i in range(N):
        y_pred, sigma_pred = GP_list[i].predict(np.array([X]).reshape(-1, 1),return_std=True)
        # print(f"y:{y_pred[0]}, y_pred: {sigma_pred}")
        Y.append(y_pred[0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

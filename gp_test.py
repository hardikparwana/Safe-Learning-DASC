import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C

noise = 0.0
kern = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-1, 10.0)) #+ WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
# kern = 1.6*ExpSineSquared(length_scale = 0.791, periodicity = 2.87)
#kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kern)#, alpha = noise, n_restarts_optimizer=10)

ypred = []
y_std = []
ytrue = []
theta = np.linspace(-np.pi,np.pi,100)
rng = np.random.RandomState(4)
X = rng.uniform(0,5,10)[:, np.newaxis]
y = np.sin(X - 2.5)**2 + 1*np.sin((X-2.5)*16)
# gp.fit(X,y)
print("y",y)
for index, t in enumerate(X):
    # y = 1*np.sin(t) #+ 1*np.sin(t*16)
    print(f"index: {index}, theta:{t} ")
    # print(f"y: {y}, t:{t}")
    # gp.fit(np.array([t]).reshape(-1, 1),np.array([y]).reshape(-1, 1))
    gp.fit(t.reshape(1,-1),y[index].reshape(1,-1))
    # ytrue.append(y)
theta = np.linspace(0,5,100)
for t in theta:
    y_pred, sigma_pred = gp.predict(np.array([t]).reshape(-1, 1),return_std=True)
    ypred.append(y_pred[0][0])
    y_std.append(sigma_pred[0])
    ytrue.append( np.sin(theta - 2.5)**2 + 1*np.sin((theta-2.5)*16) )


plt.figure()
plt.plot(X,y,'r*')
plt.plot(theta,ypred,'g')
theta = np.asarray(theta)
ypred = np.asarray(ypred)
y_std = np.asarray(y_std)
plt.fill_between(theta, ypred - y_std, ypred + y_std, alpha=0.2, color = 'm')
plt.show()
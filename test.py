from environment_models.obstacles import *
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(xlim=(-10,10),ylim=(-10,10))

obs = rectangle(0,0,1,1,ax,0)
obs2 = circle(2,0,1,ax,1)

plt.show()
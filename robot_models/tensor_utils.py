import torch
import numpy as np

def wrap_angle_tensor(angle):
        angle_numpy = angle.detach().numpy()
        factor = torch.tensor(2*np.pi,dtype=torch.float)
        if angle_numpy>np.pi:
            angle = angle - factor
        if angle<-np.pi:
            angle = angle + factor
        return angle
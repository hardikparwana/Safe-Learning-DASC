import torch
from torch.distributions import Normal
from storch.method import Reparameterization, ScoreFunction
import storch

'''
Return a storch tensor at that state
'''
class stochastci_dynamics:

    def __init__(self,mean,std):



        # Sample e from a normal distribution using reparameterization
        normal_distribution = Normal(mean, std)
        e = method(normal_distribution)






'''
TO do with the tensor:
1. combine with other tensors
2. instead of torch cost: need to do storch.add_cost(f,"f")
3. storch.backward()
'''
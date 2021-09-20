import torch

x1 = torch.ones((1,1), requires_grad=True)
x1.retain_grad()
g = torch.cos(x1[0,0])
print(x1,g)
g.backward()
print(x1.grad)

g = torch.tensor(torch.cos(x1[0,0]))
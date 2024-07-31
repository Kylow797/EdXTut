import torch 
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y, X = np.mgrid[-2.0:2.0:0.005, -2.0:2.0:0.005]
#enhanced HD version
#Y, X = np.mgrid[-0.3:0.3:0.0003, -1:0:0.0003]

x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) 
zs = z.clone() 
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

c = torch.tensor(complex(-0.123, 0.745), dtype=torch.cfloat)
#Douady set
for i in range(200):
    #Compute thenew values of z + 1 = z + x
    zs_  = zs*zs + c 
    not_diverged = torch.abs(zs_)<4.0
    ns += not_diverged
    zs = zs_


fig = plt.figure(figsize=(16,10))

def processFractal(a):
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic),155-90*np.cos(a_cyclic)], 2)
    img[a==a.max()]=0
    a = imga = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()


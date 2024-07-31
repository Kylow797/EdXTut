import torch
import numpy as np
                                                                               
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.Tensor(Y)

x = x.to(device)
y = y.to(device)

z = torch.exp(-(x**2+y**2)/2.0)
z2 = torch.sin(x+y)
pzs = z*z2

#plot 
import matplotlib.pyplot as plt

plt.imshow(pzs.cpu().numpy())

plt.tight_layout()
plt.show()


'''
Q1) Sine wave using z2
Q2) Gabor filter  using pzs
'''



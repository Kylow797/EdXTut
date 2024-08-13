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

def f(z):
    return z**4 - 1

def f_prime(z):
    return 4 * z**3

c = 1.203
#Newton set
for i in range(200):
    #Compute thenew values of z + 1 = z + f(z)/f'(z)
    zs  =  (zs - f(zs) / f_prime(zs))+1.203


roots = torch.tensor([1, -1, 1j, -1j], dtype=torch.complex64)  
distances = torch.abs(z.unsqueeze(2) - roots)
closest_root_indices = torch.argmin(distances, dim=2)

closest_root_indices_np = closest_root_indices.numpy()


fig = plt.figure(figsize=(16,10))

def processFractal(a):
    a_cyclic = (2 * np.pi * a / 4.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 90 * np.cos(a_cyclic)
    ], axis=2)
    img[a == a.max()] = 0
    a = np.uint8(np.clip(img, 0, 255))
    return a

plt.imshow(processFractal(closest_root_indices.cpu().numpy()), cmap='inferno')
plt.tight_layout(pad=0)
plt.show()


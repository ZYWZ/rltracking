from models.rltracker import build_agent
import gym
import numpy as np
import torch
import torch.nn.functional as F

tt = torch.Tensor([[531.2000], [4.1000]])
ftt = F.normalize(tt)
print(ftt)

tt = torch.Tensor([[531.2000], [4.1000]])
tt = tt.permute(1, 0)
ftt = F.normalize(tt)
ftt = ftt.permute(1, 0)
print(ftt)

tt = torch.Tensor([[5.9995e-01, 1.8131e+03, 2.2301e+03, 4.0970e+02, 3.8940e+02, 8.7950e+02,
         2.6894e+03, 2.6220e+02, 9.2880e+02, 1.3531e+03, 2.2732e+03, 7.6850e+02,
         1.9824e+03, 2.0672e+03, 1.1559e+03, 1.0677e+03, 4.5310e+02, 8.2850e+02,
         7.4070e+02, 1.1121e+03, 1.4052e+03, 1.6408e+03, 6.6530e+02, 1.2968e+03,
         5.2670e+02, 1.0405e+03]])
print(tt.shape)
ftt = F.normalize(tt)
print(ftt)
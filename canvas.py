import torch 
import numpy as np

x = torch.arange(30).view(6, 5)
print(x)

x = x.view(int(6/3), 5 * 3)  # (bsize, num_smpl_betas * 2)
print(x)
x = x.repeat_interleave(2,dim=0)
print(x)
#shape_params = shape_params.repeat_interleave(3, dim=0)
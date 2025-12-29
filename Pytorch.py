import torch
import numpy
import matplotlib.pyplot as plt
import pandas as pd

# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], # creating a tensor
                               dtype= None, # what data type is the tensor 
                               device= None, #what device is your tensor on
                               requires_grad=False, # if wished to keep track of gradients
                               )

float_16_tensor = float_32_tensor.type(dtype= torch.float16) # turn the tensors to float 16
int_32_tensor = torch.tensor([3, 6, 9],
                             dtype = torch.int32,
                             )


print(int_32_tensor * float_32_tensor) # multipilying two tensors

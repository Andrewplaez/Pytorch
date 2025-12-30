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
                             dtype = torch.int32, #intiger type
                             )

# Create a tensor 
random_tensor = torch.rand(3, 4)


#tensor operations
#" Tensor operations include "
# Addition & Subtraction
# Multiplication & division

# adding tensors
 
Adding_Tensor = torch.tensor([1, 2, 3])

print(Adding_Tensor + 10)
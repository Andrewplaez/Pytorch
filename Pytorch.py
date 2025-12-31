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

# dot product 
matrix_1 = torch.tensor([[1, 2, 3],
                         [3, 4, 5]])

matrix_2 = torch.tensor([[6, 7, 8],
                         [7, 9, 11]])

print(Adding_Tensor + 10)

# in built tensor functions 
print(torch.mul(Adding_Tensor, 10))

# Multiplying tensor
print(torch.matmul(matrix_1, matrix_2.T))

# rules for multiplying tensors
# 1 : the inner dimensions must match 
# 2 : the resulting matricx has the dimension of the outer matrix

# Shapes for matrix multiplication
#Create two Matrices 

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6],])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])


# to fix a tensor shape issues we can manipulate the shape of our tensor
#by using transpose

New_Tensor_B = tensor_B.T

# min, max, mean, sum 

torch.min(tensor_A) or tensor_A.min

#find the mean 

torch.mean(tensor_A.type(torch.float32))

# find the sum
torch.sum(tensor_A)
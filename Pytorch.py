import torch
import numpy
import matplotlib.pyplot as plt
import pandas as pd

#Scalar
scalar = torch.tensor(7)


# Vector
vector = torch.tensor([7, 5])


# Matrices
Matrix = torch.tensor([[1, 2], 
                       [3, 4]])

#create a randon tensor
random_tensor = torch.rand(1, 3, 4)

#Created a Zeros tensor
ZeroTensor = torch.zeros(1, 3, 4)

# Create all ones
Ones = torch.ones(1, 3, 4)

#create a randon range of numbers in the tensor
one_to_ten = torch.arange(start=1, end=1000, step=80)

# copying the range
Zeros = torch.zeros_like(one_to_ten)

# testing
print(Zeros)

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

# prints 
print(scalar.ndim, vector.ndim, Matrix.ndim, sep=" : ")
print(Matrix.shape, vector.shape)
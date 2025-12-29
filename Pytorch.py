import torch
import numpy
import matplotlib.pyplot as plt
import pandas as pd

#Scalar
scalar = torch.tensor(7)
print(scalar.ndim)

# Vector
vector = torch.tensor([7, 5])
print(vector.ndim)
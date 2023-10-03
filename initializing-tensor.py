import torch

# Initializing Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)  # CPU or GPU
print(my_tensor.shape)   # size
print(my_tensor.requires_grad)  # does it need gradient or not?

# Other common initialization methods
a = torch.empty(size=(3, 3))
b = torch.zeros((3, 3))
c = torch.rand((3, 3))  # tensor with random numbers
d = torch.ones((3, 3))  # all of the values = 1
e = torch.eye(5, 5)    # main diameter values = 1
f = torch.arange(start=0, end=5, step=1)  # tensor([0, 1, 2, 3, 4])
g = torch.linspace(start=0.1, end=1, steps=4)  # tensor([0.1000, 0.4000, 0.7000, 1.0000])
h = torch.diag(torch.ones(3))   # main diameter values = 1

print(a,b,c,d,e,f,g,h)

# convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())  # convert to bool
print(tensor.short())  # convert to int16
print(tensor.long())  # convert to int64   (important)
print(tensor.half())  # convert to float16
print(tensor.float())  # convert to float32  (important)
print(tensor.double())  # convert to float64

# array to tensor conversion
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
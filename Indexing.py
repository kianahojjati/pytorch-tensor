import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  #first row

x[0,0] = 100

print(x[:, 0].shape)  #first column

print(x[2, 0:10])  # first 10 columns of the third row

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])  # just print the elements in the specified indices

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])  # x[1,4] x[0,0]

#advanced indexing
x = torch.arange(10)
print(x[(x< 2) | (x> 8)])  # x[0] x[1] x[9]
print(x[x.remainder(2) == 0]) # x[0] x[2] x[4] ...

#useful operations
print(torch.where(x>5, x, x*2)) #if(x>5) return x , else return x*2
print(x.ndimension())  # number of dimensions
print(x.numel())  # number of elements in x
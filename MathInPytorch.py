import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Add:
z2 = torch.add(x, y)
z = x + y

# Sub:
z1 = x-y

# Div:
z3 = torch.true_divide(x, y)

# inplace operations:
t = torch.zeros(3)
t.add_(x)

# Power:
z4 = x ** 2

# Matrix Mul :
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3

# matrix power :
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise mult.
z5 = x * y  # 9,16,21

# dot product:
z6 = torch.dot(x, y) # 46

# exmaple of broadcasting:
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z7 = x1 - x2

# other useful operations:
values, indices = torch.max(x, dim=0)   # or min
abs_x = torch.abs(x)
z8 = torch.eq(x, y) # compare every element
torch.sort(y, dim=0, descending=False)

z9 = torch.clamp(x, min=0) # ReLU func.
from __future__ import print_function
import torch



## Construct a 5x3 matrix, uninitialized:
x = torch.empty(5, 3)
print(x)

## Construct a randomly initialized matrix:
x = torch.rand(5,3)
print(x)

## Construct a matrix filled zeros and of dtype long:

x = torch.zeros(5,3,dtype=torch.long)
print(x)

## Construct a tensor directly from data:

x = torch.tensor([5.5,3])
print(x)

## create a tensor based on an existing tensor

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

## Get its size:

print(x.size())

## Addition: syntax 1

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)


## slicing
print(x[:, 1])

## resizing
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


## get value

x = torch.randn(1)
print(x)
print(x.item())

## Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

## value changes
a.add_(1)
print(a)
print(b)


## Converting NumPy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

## To check how many CUDA supported GPU’s are connected to the machine
print(torch.cuda.device_count())

## get the name of the GPU Card connected to the machine
print(torch.cuda.get_device_name(0))

## CUDA Tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!



import torch

## Create a tensor and set requires_grad=True to track computation with it

x = torch.ones(2, 2, requires_grad=True)
print(x)

## Do a tensor operation:

y = x + 2
print("Y: ")
print(y)
print(y.grad_fn)
print("************")


z = y * y * 3
out = z.mean()

print("z,out: ")
print(z, out)
print("************")

## Gradients z = 3* (x+2)^2
out.backward()
print(x.grad)

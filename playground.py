import torch

x = torch.tensor([25.0])
x = x.unsqueeze(0)
x = x.squeeze()
print(x.shape)

x = torch.tensor([1,2,3,4,5])

print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)

zeros = torch.zeros((3,4))
print("ZEROS TENSOR:", zeros)
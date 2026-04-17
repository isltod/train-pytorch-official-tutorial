import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand}\n")

# 여기까지 와서 보니 torch = np 로 생각하면 되겠네...그럼 torch as th는 어떤가?
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

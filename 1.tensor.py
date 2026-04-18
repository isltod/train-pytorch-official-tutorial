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

# 텐서의 대표적 속성은 shape, type, 저장 장소
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# GPU가 존재하면 텐서를 이동...간단하게 to
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# Numpy 방식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
# 이렇게 가로든 세로든 한 줄을 뽑아내면 다 (1,n)이 된다...
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 텐서 이어붙이기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
# 일단 stack은 새 축을 만들고 그 방향으로 쌓는거 같은데...
t2 = torch.stack([tensor, tensor, tensor])
print(t2)

# 산술 연산...
# 이쪽이 행렬 곱인데...
# @ 연산자는 행렬곱 matmul
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
print("y3 tmp:", y3)
# matmul에 입력 둘 출력 다 지정할 수도...
torch.matmul(tensor, tensor.T, out=y3)
print("y1:", y1)
print("y2:", y2)
print("y3:", y3)

# 위와 같은 방식의 아다마르 연산자...뒤들 전치하지 않는다...
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print("z1:", z1)
print("z2:", z2)
print("z3:", z3)

# 단일 요소 텐서 - 원소가 하나라도 자동으로 스칼라 되는게 아니고 item() 사용...
agg = tensor.sum()
agg_item = agg.item()
print(agg, type(agg))
print(agg_item, type(agg_item))

# in-place, 바꿔치기 연산 - 뒤에 _ 붙인 연산자...
print(f"{tensor} \n")
tensor.add(5)
print("after add:", tensor)
tensor.add_(5)
print("after add_:", tensor)
# 이건 미분 계산에 문제가 있을 수 있어 권장하지 않는다? 그럼 왜 가르쳐주나?

# Numpy 변환
t = torch.ones(5)
print(f"t: {t}")
# 이렇게 하면 넘파이로(to는 GPU) 가는데...참조 방식이라 하나를 건드리면 다른 것도 변한다...
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
# 넘파이에서 텐서는 from_numpy, 이 쪽도 참조 방식이다...
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
#

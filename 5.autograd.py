import torch

x = torch.ones(5)
y = torch.zeros(3)
# 역전파로 최적화할 대상은 자동 미분 설정
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 매개변수(최종 결과) 없이 호출하면 backward(torch.tensor(1.0))
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(4, 5, requires_grad=True)
# 각 원소에 +1
a = inp + 1
# 각 원소를 제곱
b = a.pow(2)
# 전치
c = b.t()
out = (inp + 1).pow(2).t()
# 일단...backward 두 번 이상 호출하려면 retain_graph=True, 그런데 누적 안되려면 grad.zero_
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradien\n{inp.grad}")

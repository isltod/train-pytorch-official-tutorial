import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # 이런 식으로 층을 순서대로 쌓아놓고 한 번에 통과시키는 모양...
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # (1,28,28)을 (1,784)로 변경 - 1차원화? dim=0 배치 차원은 항상 유지되나보다...
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 모델을 만들어서 cuda로 옮기고...
model = NeuralNetwork().to(device)
print(model)

# cuda에 텐서 만들고 forward...
X = torch.rand(1, 28, 28, device=device)
# forward 직접 호출 금지...
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 각각...
input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
# 여기서는 (3, 28, 28) -> (3, 784)
flat_image = flatten(input_image)
print(flat_image.size())

# 선형변환 xW+b=y, (3,784)x(784,20)
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# ReLU 활성화 - 텐서들 안에서 0이하는 0, 아니면 원래 숫자로 뱉기...
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequential로 이어 붙이기...
# (3,28,28) -> f -> (3,784) -> l1 -> (3,20) -> r -> (3,20) -> l2 -> (3,10) 이런 식이어야 하는데...
# 그럼 안 넣어도 알아서 l1을 (784,20)으로 맞춰주나?
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits)

# 소프트맥스에서 dim은 합이 1이 되는 축...즉 변수 확률 분포 축...
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)

# 모델 매개변수 접근
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

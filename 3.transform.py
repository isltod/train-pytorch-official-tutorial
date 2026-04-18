import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # PIL Image나 ndarray를 실수 텐서로 변환
    transform=ToTensor(),
    #
    target_transform=Lambda(
        # 10개 0 텐서 만들고, y 자리에 1
        # scatter_: scatter의 in-place 버전
        # 0축(행)으로 뿌리는데, 텐서로 주는 인덱스들 위치에, value 값을 넣는다...
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)
# 이건 데이터셋 정의
# print(ds)
# 이건 데이터와 라벨 들어있는 튜플
ds_0 = ds[0]
print(type(ds_0))
# 이게 텐서, 데이터 1개
ds_0_0 = ds_0[0]
print(type(ds_0_0))
print(ds[0][0].ndim)
print(ds[0][0].shape)
# 이건 라벨 텐서 1개
ds_0_1 = ds_0[1]
print(type(ds_0_1))
print(ds[0][1])
print(ds[0][1].ndim)
print(ds[0][1].shape)

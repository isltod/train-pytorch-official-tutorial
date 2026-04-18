import torch
import torchvision.models as models

# IMAGENET1K_V1이라는 가중치를 불러와서 만든 모델...저장...
model1 = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model1.state_dict(), "model_weights.pth")

model2 = models.vgg16()  # 기본 가중치를 불러오지 않으므로 학습 안된 모델인데...
model2.load_state_dict(torch.load("model_weights.pth"))
# 쓰기 전에 eval을 해야 드롭아웃과 배치정규화를 평가모드로 설정하고, 그래야 일관성 있게 추론한다고?
model2.eval()

# 모델 자체를 가중치와 같이 저장하는 방법도...엥? 오륜데? 튜토리얼도 오류라고...뭐냐 이게?
torch.save(model1, "model.pth")
model3 = torch.load("model.pth")
model3.eval()

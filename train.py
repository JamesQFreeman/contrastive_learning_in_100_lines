import torch
from BYOL import BYOL
from torchvision import models

resnet = models.resnet50()
learner = BYOL(net = resnet)
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_random_images():
    return torch.randn(20,3,224,224)

for epoch in range(5):
    for _ in range(100):
        images = sample_random_images()
        loss = learner(images)
        print(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
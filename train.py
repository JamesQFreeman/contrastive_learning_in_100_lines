import torch
from BYOL import BYOL
from torchvision import models
from dataset import ImageNet_5Class

resnet = models.resnet50()
learner = BYOL(net=resnet)

my_dataset = ImageNet_5Class()
trainloader = torch.utils.DataLoader(my_dataset, batch_size=128, suffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

# loop over the dataset multiple times
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = learner(inputs)
        loss.backward()
        optimizer.step()
        learner.update_moving_average()

        running_loss += loss.item()

    print('Loss: {}'.format(running_loss))

print('Finished Training')

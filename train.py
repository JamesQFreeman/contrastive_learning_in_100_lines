import torch
from BYOL import BYOL
from torchvision import models
from dataset import ImageNet_5Class


def self_supervise_train(n_epoch: int = 5):
    # Can be pretrained=True if self-supervised on IN pretrained weight is prefered
    resnet = models.resnet50()
    # Optimal hyperparameter is set according to paper. Change that in model file
    learner = BYOL(net=resnet)

    my_dataset = ImageNet_5Class(augmentation=False, annotation=False)
    trainloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    learner.to(device)
    # loop over the dataset/multiple times
    for _ in range(n_epoch):
        running_loss = 0.0
        for _, data in enumerate(trainloader, 0):
            inputs = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = learner(inputs)
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            running_loss += loss.item()

        print('Loss: {}'.format(running_loss))

    torch.save(learner.online_encoder.state_dict(), "encoder.pth")
    print('Finished Training')


self_supervise_train()

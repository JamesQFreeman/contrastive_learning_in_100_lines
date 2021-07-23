import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset import ImageNet_5Class
from torch.utils.data import DataLoader


def new_head(n_classes: int, net):
    n_features = net.fc.in_features
    head = nn.Linear(n_features, n_classes)
    net.fc = head
    return net


def get_IN_pretrain():
    """weight from ImageNet"""
    IN_pretrained_net = models.resnet50(pretrained=True)
    return new_head(5, IN_pretrained_net)


def get_BYOL_pretrain():
    """weight from BYOL"""
    print(torch.load('encoder.pth').keys())
    BYOL_pretrained_net = models.resnet50(pretrained=False)
    BYOL_pretrained_net.load_state_dict(
        torch.load("encoder.pth"), strict=False)
    return new_head(5, BYOL_pretrained_net)


def train(net, device, n_epoch, trainloader, testloader):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    # loop over the dataset multiple times
    for epoch in range(n_epoch):
        running_loss = 0.0

        # Train
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Loss: {}'.format(running_loss))

        # Test
        with torch.no_grad():
            total, correct = 0, 0
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(
                f'Accuracy of the network on the test images: {100 * correct / total} %')

    print('Finished Training')


if __name__ == "__main__":
    train_dataset = ImageNet_5Class(
        train=True, augmentation=True, annotation=True)
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = ImageNet_5Class(
        train=False, augmentation=True, annotation=True)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_BYOL_pretrain()
    net.to(device)
    train(net, device, 10, trainloader, testloader)

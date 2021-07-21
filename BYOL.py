import torch
from torch import nn

from augmentation import SimCLR_augment


def get_encoder(net: nn.Module) -> nn.Module:
    """ input a network and output it's convolutional feature encoder"""
    return nn.Sequential(*(list(net.children())[:-1]))


def loss_fn(x, y):
    x = nn.functional.normalize(x, dim=-1, p=2)
    y = nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2*(x*y).sum(dim=-1)


def MLP(in_size, out_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, out_size)
    )


def EMA(moving_average_model: nn.Module, current_model: nn.Module, beta: float):
    """Exponential Moving Average"""
    for c_params, ma_params in zip(current_model.parameters(), moving_average_model.parameters()):
        ma_weight, c_weight = ma_params.data, c_params.data
        ma_params.data = beta*ma_weight + (1-beta)*c_weight


class BYOL(nn.Module):
    def __init__(self, net: nn.Module, moving_average_decay: float = 0.99) -> None:
        super().__init__()
        num_features = net.fc.in_features
        print(num_features, "num features\n")
        self.augment1 = SimCLR_augment
        self.augment2 = SimCLR_augment
        self.target_encoder = get_encoder(net)
        self.online_encoder = get_encoder(net)  # they have same weight
        self.target_projector = MLP(in_size=num_features, out_size=256)
        self.online_projector = MLP(in_size=num_features, out_size=256)

        # use EMA to copy weight of target to online for initialization
        EMA(self.target_encoder, self.online_encoder, beta=1)
        EMA(self.target_projector, self.online_projector, beta=1)

        self.online_predictor = MLP(in_size=256, out_size=256)
        self.moving_average_decay = moving_average_decay

    def update_moving_average(self):
        EMA(self.target_encoder, self.online_encoder, self.moving_average_decay)
        EMA(self.target_projector, self.online_projector, self.moving_average_decay)

    def online_pipeline(self, x):
        return self.online_predictor(self.online_projector(torch.flatten(self.online_encoder(x), 1)))

    def target_pipeline(self, x):
        return self.target_projector(torch.flatten(self.target_encoder(x), 1))

    def forward(self, x):
        view1, view2 = self.augment1(x), self.augment2(x)
        pred1, pred2 = self.online_pipeline(view1), self.online_pipeline(view2)
        with torch.no_grad():
            proj1, proj2 = self.target_pipeline(
                view1), self.target_pipeline(view2)
        loss = loss_fn(pred1, proj2.detach()) + loss_fn(pred2, proj1.detach())
        return loss.mean()


if __name__ == "__main__":
    from torchvision.models import resnet50
    net = resnet50()
    my_model = BYOL(net)
    rand_input = torch.randn(2, 3, 224, 224)
    my_model.online_encoder(rand_input)
    loss = my_model(rand_input)
    print(loss)

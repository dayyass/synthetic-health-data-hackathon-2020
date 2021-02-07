import torch.nn as nn
from torchvision.models import resnet18


class ConvDropoutLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.activation(self.dropout(self.conv(x))))


class ConvBatchNormLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.activation(self.batch_norm(self.conv(x))))


class LinearDropoutLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dropout(self.linear(x)))


class LinearBatchNormLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = nn.BatchNorm1d(num_features=out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.linear(x)))


class ConvDropoutNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        conv_dropout: float = 0.0,
        linear_dropout: float = 0.0,
    ):
        super().__init__()

        # convolutional layers
        self.conv_1 = ConvDropoutLayer(
            in_channels=in_channels, out_channels=8, kernel_size=3, dropout=conv_dropout
        )
        self.conv_2 = ConvDropoutLayer(
            in_channels=8, out_channels=16, kernel_size=3, dropout=conv_dropout
        )
        self.conv_3 = ConvDropoutLayer(
            in_channels=16, out_channels=32, kernel_size=3, dropout=conv_dropout
        )
        self.conv_4 = ConvDropoutLayer(
            in_channels=32, out_channels=64, kernel_size=3, dropout=conv_dropout
        )
        self.conv_5 = ConvDropoutLayer(
            in_channels=64, out_channels=128, kernel_size=3, dropout=conv_dropout
        )

        # linear layers
        self.linear_1 = LinearDropoutLayer(
            in_features=128 * 2 * 2, out_features=128, dropout=linear_dropout
        )
        self.linear_2 = LinearDropoutLayer(
            in_features=128, out_features=64, dropout=linear_dropout
        )
        self.linear_3 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x


class ConvBatchNormNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        # convolutional layers
        self.conv_1 = ConvBatchNormLayer(
            in_channels=in_channels, out_channels=8, kernel_size=3
        )
        self.conv_2 = ConvBatchNormLayer(in_channels=8, out_channels=16, kernel_size=3)
        self.conv_3 = ConvBatchNormLayer(in_channels=16, out_channels=32, kernel_size=3)
        self.conv_4 = ConvBatchNormLayer(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_5 = ConvBatchNormLayer(
            in_channels=64, out_channels=128, kernel_size=3
        )

        # linear layers
        self.linear_1 = LinearBatchNormLayer(in_features=128 * 2 * 2, out_features=128)
        self.linear_2 = LinearBatchNormLayer(in_features=128, out_features=64)
        self.linear_3 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x


def make_resnet18(num_classes: int, one_channel_input: bool = False):
    model = resnet18(num_classes=num_classes)

    if one_channel_input:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    return model


def make_pretrained_resnet18(num_classes: int, one_channel_input: bool = False):
    model = resnet18(pretrained=True)

    # average channel params
    if one_channel_input:
        avg_channel_weight = model.conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.conv1.weight.data = avg_channel_weight

    if num_classes != 1000:
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    return model

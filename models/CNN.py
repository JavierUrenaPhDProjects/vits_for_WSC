import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, channels=3, num_classes=1, depth='base'):
        super(CNN, self).__init__()

        # Define layers based on the depth
        if depth == 'base':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            feature_dim = 16 * 192 * 192  # Adjust based on the output size of the conv layer
        elif depth == 'medium':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            feature_dim = 64 * 96 * 96  # Adjust based on the output size of the conv layer
        elif depth == 'deep':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            feature_dim = 256 * 48 * 48  # Adjust based on the output size of the conv layer
        else:
            raise ValueError("Unsupported depth")

        # Nonlinear regressor to map from feature dimensions to num_classes
        self.regressor = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten the features for the regressor
        x = self.regressor(x)
        return x


def CNN_base(args):
    return CNN(channels=args['channels'], num_classes=1, depth='base')


def CNN_medium(args):
    return CNN(channels=args['channels'], num_classes=1, depth='medium')


def CNN_deep(args):
    return CNN(channels=args['channels'], num_classes=1, depth='deep')


if __name__ == '__main__':
    # Test the model
    x = torch.randn(1, 3, 384, 384)
    model = CNN_deep({'channels': 3})
    print(sum(p.numel() for p in model.parameters()) / 1e6)

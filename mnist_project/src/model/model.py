# src/model/model.py
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super(ConvNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=n_kernels*2, out_channels=n_kernels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(in_features=n_kernels*4*3*3, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=128, out_features=output_size)
        )

    def forward(self, x):
        return self.net(x)

class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        self.net = nn.Sequential()
    def forward(self, x):
        return self.net(x)
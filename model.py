import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, depth= 1):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 16*depth, kernel_size= 3),
            nn.BatchNorm2d(num_features= 16*depth),
            nn.ReLU(),
            nn.Conv2d(in_channels= 16*depth, out_channels= 32*depth, kernel_size= 3),
            nn.BatchNorm2d(num_features= 32*depth),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32*depth, out_channels= 64*depth, kernel_size= 3),
            nn.BatchNorm2d(num_features= 64*depth),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64*depth, out_channels= 128*depth, kernel_size= 3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels= 128*depth, out_channels= 64*depth, kernel_size= 3),
            nn.BatchNorm2d(64*depth),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels= 64*depth, out_channels= 32*depth, kernel_size= 3),
            nn.BatchNorm2d(32*depth),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels= 32*depth, out_channels= 16*depth, kernel_size= 3),
            nn.BatchNorm2d(16*depth),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels= 16*depth, out_channels= 1, kernel_size= 3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

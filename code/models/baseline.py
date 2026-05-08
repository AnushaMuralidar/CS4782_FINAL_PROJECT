import torch
import torch.nn as nn

# MODEL
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch)
        )

    def forward(self, x):
        return x + self.net(x)


class DLocBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.Tanh(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            *[ResBlock(256) for _ in range(6)]
        )

        # LOCATION DECODER
        self.loc_decoder = nn.Sequential(
            *[ResBlock(256) for _ in range(3)],

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # CONSISTENCY DECODER
        self.cons_decoder = nn.Sequential(
            *[ResBlock(256) for _ in range(6)],

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 4, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.encoder(x)
        loc  = self.loc_decoder(feat)
        cons = self.cons_decoder(feat)
        return loc, cons
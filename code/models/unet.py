import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDoubleConv(nn.Module):
    """Two conv layers with a residual skip."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
        )
        # 1×1 projection so skip has same channels as output
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


#  UNet model 

class DLocUNet(nn.Module):
    def __init__(self, in_ch=4, base=64):
        super().__init__()

        #  Encoder 
        self.enc1 = ResDoubleConv(in_ch,    base)        # [B, 64,  H,   W  ]
        self.enc2 = ResDoubleConv(base,     base*2)      # [B, 128, H/2, W/2]
        self.enc3 = ResDoubleConv(base*2,   base*4)      # [B, 256, H/4, W/4]
        self.enc4 = ResDoubleConv(base*4,   base*8)      # [B, 512, H/8, W/8]
        self.pool = nn.MaxPool2d(2)

        #  Bottleneck (deeper = more context) 
        self.bottleneck = nn.Sequential(
            ResDoubleConv(base*8, base*16),
            ResDoubleConv(base*16, base*16),
            ResDoubleConv(base*16, base*16),
            nn.Dropout2d(0.3),
        )                                                # [B,1024, H/16,W/16]

        # Shared decoder trunk (4 upsampling stages) 
        self.dec4 = ResDoubleConv(base*16 + base*8,  base*8)
        self.dec3 = ResDoubleConv(base*8  + base*4,  base*4)
        self.dec2 = ResDoubleConv(base*4  + base*2,  base*2)
        self.dec1 = ResDoubleConv(base*2  + base,    base)

        #  Location head (outputs sparse probability map)
        self.loc_head = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 1, 1),
            nn.Sigmoid(),
        )

        #  Consistency head (outputs offset-corrected heatmaps, one per AP) 
        self.cons_head = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 4, 1),
            nn.Tanh(),
        )

    def _up_cat(self, x, skip):
        """Upsample x to match skip's spatial size, then concat."""
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        # Decoder (single pass — shared features)
        d4 = self.dec4(self._up_cat(b,  e4))
        d3 = self.dec3(self._up_cat(d4, e3))
        d2 = self.dec2(self._up_cat(d3, e2))
        d1 = self.dec1(self._up_cat(d2, e1))

        loc  = self.loc_head(d1)    # [B, 1, H, W]
        cons = self.cons_head(d1)   # [B, 4, H, W]
        return loc, cons
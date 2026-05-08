import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w
    

class ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se   = SEBlock(out_ch)   

    def forward(self, x):
        out = self.conv(x) + self.skip(x)
        out = F.relu(out, inplace=True)
        out = self.se(out)   
        return out
    
class TransformerBottleneck(nn.Module):
    
    def __init__(self, channels, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.conv_in = ResDoubleConv(channels, channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = channels,
            nhead      = nhead,
            dim_feedforward = channels * 4,
            dropout    = dropout,
            batch_first = True,        # [B, seq, dim]
            norm_first  = True         # pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.conv_out = ResDoubleConv(channels, channels)
        self.drop     = nn.Dropout2d(0.3)

    def forward(self, x):
        # x: [B, C, H, W]
        x   = self.conv_in(x)
        B, C, H, W = x.shape
        
        # Flatten spatial  sequence: [B, H*W, C]
        seq = x.flatten(2).permute(0, 2, 1)
        
        # Self-attention across all spatial positions
        seq = self.transformer(seq)               # [B, H*W, C]
        
        # Reshape back to feature map: [B, C, H, W]
        out = seq.permute(0, 2, 1).view(B, C, H, W)
        out = self.conv_out(out)
        return self.drop(out)


class DLocUNetTransformer(nn.Module):
    def __init__(self, in_ch=4, base=64):
        super().__init__()

        # Encoder (same as UNet v2)
        self.enc1 = ResDoubleConv(in_ch,    base)
        self.enc2 = ResDoubleConv(base,     base*2)
        self.enc3 = ResDoubleConv(base*2,   base*4)
        self.enc4 = ResDoubleConv(base*4,   base*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck: conv to expand channels, then transformer
        self.bottleneck_conv = ResDoubleConv(base*8, base*16)
        self.bottleneck_transformer = TransformerBottleneck(
            channels   = base*16,   # 1024
            nhead      = 8,
            num_layers = 2,         # 2 transformer layers
            dropout    = 0.1
        )

        # Decoder (same as UNet v2)
        self.dec4 = ResDoubleConv(base*16 + base*8,  base*8)
        self.dec3 = ResDoubleConv(base*8  + base*4,  base*4)
        self.dec2 = ResDoubleConv(base*4  + base*2,  base*2)
        self.dec1 = ResDoubleConv(base*2  + base,    base)

        # Output heads (same as UNet v2)
        self.loc_head = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 1, 1),
            nn.Sigmoid(),
        )
        self.cons_head = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 4, 1),
            nn.Tanh(),
        )

    def _up_cat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck: conv expand → transformer global attention
        b  = self.bottleneck_conv(self.pool(e4))
        b  = self.bottleneck_transformer(b)

        d4 = self.dec4(self._up_cat(b,  e4))
        d3 = self.dec3(self._up_cat(d4, e3))
        d2 = self.dec2(self._up_cat(d3, e2))
        d1 = self.dec1(self._up_cat(d2, e1))

        return self.loc_head(d1), self.cons_head(d1)
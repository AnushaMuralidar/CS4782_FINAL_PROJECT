import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
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
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
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
        return self.se(out)



class CrossAPAttention(nn.Module):
    """
    Cross-access-point attention applied per encoder scale.

    Args:
        channels  – number of feature-map channels at this scale
        num_aps   – number of access points (== number of input channels, 4)
        nhead     – attention heads; must divide (channels // num_aps)
    """
    def __init__(self, channels: int, num_aps: int = 4, nhead: int = 4):
        super().__init__()
        self.num_aps   = num_aps
        self.chunk     = channels // num_aps   # feature dim per AP token

        safe_nhead = nhead if self.chunk % nhead == 0 else 1

        self.attn = nn.MultiheadAttention(
            embed_dim   = self.chunk,
            num_heads   = safe_nhead,
            batch_first = True,
            dropout     = 0.0,
        )
        self.norm = nn.LayerNorm(self.chunk)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, H, W]   where C = num_aps * chunk
        returns: [B, C, H, W]  with cross-AP context blended in
        """
        B, C, H, W = x.shape

        x_aps  = x.view(B, self.num_aps, self.chunk, H, W)

        tokens = x_aps.mean(dim=[-2, -1])

        attended, _ = self.attn(tokens, tokens, tokens)
        attended    = self.norm(attended + tokens)

        scale = attended.view(B, self.num_aps, self.chunk, 1, 1)

        scale_full = scale.expand_as(x_aps).reshape(B, C, H, W)

        return x + self.gate * scale_full


def make_2d_sinusoidal_pe(H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
    """
    Returns a positional encoding tensor of shape [1, H*W, C].

    Half the channels encode row position, half encode column position,
    both using the standard sinusoidal schedule from "Attention is All You Need".
    """
    half = C // 2
    div  = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / half)
    )

    rows = torch.arange(H, dtype=torch.float32, device=device)   # [H]
    cols = torch.arange(W, dtype=torch.float32, device=device)   # [W]

    pe_row = torch.zeros(H, half, device=device)
    pe_row[:, 0::2] = torch.sin(rows.unsqueeze(1) * div.unsqueeze(0))
    pe_row[:, 1::2] = torch.cos(rows.unsqueeze(1) * div.unsqueeze(0))

    # Col encoding: [W, half]
    pe_col = torch.zeros(W, half, device=device)
    pe_col[:, 0::2] = torch.sin(cols.unsqueeze(1) * div.unsqueeze(0))
    pe_col[:, 1::2] = torch.cos(cols.unsqueeze(1) * div.unsqueeze(0))

    pe_row_grid = pe_row.unsqueeze(1).expand(H, W, half)
    pe_col_grid = pe_col.unsqueeze(0).expand(H, W, half)

    pe = torch.cat([pe_row_grid, pe_col_grid], dim=-1) 
    return pe.view(1, H * W, C)


class TransformerBottleneck(nn.Module):
    """
    Transformer bottleneck with 2-D sinusoidal positional encoding.

    The positional encoding is added to the token sequence before the
    transformer layers so the attention mechanism knows the spatial origin
    of each token.
    """
    def __init__(self, channels: int, nhead: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv_in  = ResDoubleConv(channels, channels)
        self.conv_out = ResDoubleConv(channels, channels)
        self.drop     = nn.Dropout2d(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = channels,
            nhead           = nhead,
            dim_feedforward = channels * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        self._pe_cache: dict = {}

    def _get_pe(self, H: int, W: int, C: int,
                device: torch.device) -> torch.Tensor:
        key = (H, W, C, str(device))
        if key not in self._pe_cache:
            self._pe_cache[key] = make_2d_sinusoidal_pe(H, W, C, device)
        return self._pe_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x           = self.conv_in(x)
        B, C, H, W  = x.shape

        seq = x.flatten(2).permute(0, 2, 1)

        #Add 2-D positional encoding
        # pe : [1, H*W, C] → broadcast over batch
        pe  = self._get_pe(H, W, C, x.device)
        seq = seq + pe

        seq = self.transformer(seq)

        out = seq.permute(0, 2, 1).view(B, C, H, W)
        return self.drop(self.conv_out(out))


class DLoc_APAttn_PE(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 64):
        super().__init__()

        self.enc1 = ResDoubleConv(in_ch,  base)
        self.enc2 = ResDoubleConv(base,   base * 2)
        self.enc3 = ResDoubleConv(base*2, base * 4)
        self.enc4 = ResDoubleConv(base*4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.ap_attn1 = CrossAPAttention(base,    num_aps=4, nhead=4)
        self.ap_attn2 = CrossAPAttention(base*2,  num_aps=4, nhead=4)
        self.ap_attn3 = CrossAPAttention(base*4,  num_aps=4, nhead=4)

        self.bottleneck_conv        = ResDoubleConv(base*8, base*16)
        self.bottleneck_transformer = TransformerBottleneck(
            channels   = base * 16,
            nhead      = 8,
            num_layers = 2,
            dropout    = 0.1,
        )

        self.dec4 = ResDoubleConv(base*16 + base*8,  base * 8)
        self.dec3 = ResDoubleConv(base*8  + base*4,  base * 4)
        self.dec2 = ResDoubleConv(base*4  + base*2,  base * 2)
        self.dec1 = ResDoubleConv(base*2  + base,    base)

        self.loc_head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.cons_head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 4, 1),
            nn.Tanh(),
        )

    def _up_cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor):
        e1 = self.ap_attn1(self.enc1(x))
        e2 = self.ap_attn2(self.enc2(self.pool(e1)))
        e3 = self.ap_attn3(self.enc3(self.pool(e2)))
        e4 =               self.enc4(self.pool(e3))   # no AP attn here

        b = self.bottleneck_conv(self.pool(e4))
        b = self.bottleneck_transformer(b)

        d4 = self.dec4(self._up_cat(b,  e4))
        d3 = self.dec3(self._up_cat(d4, e3))
        d2 = self.dec2(self._up_cat(d3, e2))
        d1 = self.dec1(self._up_cat(d2, e1))

        return self.loc_head(d1), self.cons_head(d1)
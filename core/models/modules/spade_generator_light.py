# coding: utf-8
"""
Light version of Spade decoder(G).
- Keep variable names identical to spade_generator.py for minimal integration changes.
- Differences are annotated with "LIGHT:" comments.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .util import SPADEResnetBlock  # same dependency as original


class _SPADEPass(nn.Module):
    """LIGHT: skip block (acts like Identity but keeps (x, seg) signature)."""
    def forward(self, x, seg):
        return x


class SPADEDecoderLight(nn.Module):
    def __init__(
        self,
        upscale=2,
        max_features=512,
        block_expansion=64,
        out_channels=48,          # LIGHT: was 64 in original
        num_down_blocks=2,

        # LIGHT knobs
        hidden_mult=0.75,         # 512 -> 384 when input_channels=256
        mid_mult=0.75,            # 256 -> 192
        num_middle_blocks=4       # LIGHT: use only first N middle blocks, skip the rest
    ):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()

        norm_G = "spadespectralinstance"
        label_num_channels = input_channels  # original uses feature itself as seg: Bx256x64x64

        # ===== channel plan (must accept input feature: Bx256x64x64) =====
        # original: 256 -> 512
        hidden_channels = int((2 * input_channels) * hidden_mult)   # LIGHT: 512 -> 384 (if mult=0.75)
        mid_channels = int(input_channels * mid_mult)               # LIGHT: 256 -> 192 (if mult=0.75)

        # LIGHT: narrower fc (keeps name self.fc)
        self.fc = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)

        # ----- G_middle blocks (names identical) -----
        # original: 6 middle blocks all at 512 channels
        # LIGHT: keep first num_middle_blocks as real blocks, rest become _SPADEPass()
        self.G_middle_0 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)

        # LIGHT: skip remaining blocks by default
        self.G_middle_4 = _SPADEPass()
        self.G_middle_5 = _SPADEPass()

        # If you want 3 or 4 middle blocks, swap pass blocks to real blocks here:
        if num_middle_blocks >= 5:
            self.G_middle_4 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)
        if num_middle_blocks >= 6:
            self.G_middle_5 = SPADEResnetBlock(hidden_channels, hidden_channels, norm_G, label_num_channels)

        # ----- up blocks (names identical) -----
        # original:
        #   up_0: 512 -> 256
        #   up_1: 256 -> 64
        # LIGHT:
        #   up_0: hidden_channels -> mid_channels
        #   up_1: mid_channels -> out_channels (smaller than 64)
        self.up_0 = SPADEResnetBlock(hidden_channels, mid_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(mid_channels, out_channels, norm_G, label_num_channels)

        # same as original
        self.up = nn.Upsample(scale_factor=2)

        # same output head logic (upscale=2 makes 512x512 via PixelShuffle)
        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
            )

    def forward(self, feature):
        # same as original: seg is the feature itself
        seg = feature  # Bx256x64x64  (original comment)

        x = self.fc(feature)

        # same call sites / variable names as original
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.sigmoid(x)
        return x

    def load_model(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        self.eval()
        return self

# dense_motion_light.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 기존 dense_motion.py가 쓰는 유틸들 그대로 사용
# (패키지 경로는 너 프로젝트 구조에 맞춰 조정)
from .util import kp2gaussian, make_coordinate_grid


# -----------------------------
# (2+1)D Conv: Conv3D를 공간/시간으로 분해
#   Conv3d(k,k,k) ~= Conv3d(1,k,k) -> Conv3d(k,1,1)
# -----------------------------
class Conv2Plus1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=False,
        mid_channels=None,
    ):
        super().__init__()
        k = kernel_size
        p = padding
        if mid_channels is None:
            # 보통 out_channels로 두는 게 무난
            mid_channels = out_channels

        # spatial: (1, k, k)
        self.conv_spatial = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(1, k, k),
            padding=(0, p, p),
            bias=bias,
        )
        # temporal: (k, 1, 1)
        self.conv_temporal = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(k, 1, 1),
            padding=(p, 0, 0),
            bias=bias,
        )

    def forward(self, x):
        x = self.conv_spatial(x)
        x = self.conv_temporal(x)
        return x


class UpBlock3d_2p1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super().__init__()
        self.conv = Conv2Plus1D(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        # 기존 util.UpBlock3d와 동일한 업샘플 방식
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class DownBlock3d_2p1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super().__init__()
        self.conv = Conv2Plus1D(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class EncoderLight(nn.Module):
    """
    Hourglass Encoder (light)
    - num_blocks 감소 가능
    - max_features 감소 가능
    - DownBlock3d를 (2+1)D conv로 교체
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        down_blocks = []
        for i in range(num_blocks):
            in_ch = in_features if i == 0 else min(max_features, block_expansion * (2 ** i))
            out_ch = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock3d_2p1D(in_ch, out_ch, kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for blk in self.down_blocks:
            outs.append(blk(outs[-1]))
        return outs


class DecoderLight(nn.Module):
    """
    Hourglass Decoder (light)
    - num_blocks 감소 가능
    - max_features 감소 가능
    - UpBlock3d를 (2+1)D conv로 교체
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d_2p1D(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

        # 마지막 conv도 2+1D로 교체 (가벼워짐)
        self.conv = Conv2Plus1D(self.out_filters, self.out_filters, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class HourglassLight(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = EncoderLight(block_expansion, in_features, num_blocks, max_features)
        self.decoder = DecoderLight(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------
# DenseMotionNetworkLight:
# dense_motion.py의 DenseMotionNetwork와 "입출력/forward" 동일 유지
# 단, hourglass만 light 버전으로 교체
# ---------------------------------------------------------
class DenseMotionNetworkLight(nn.Module):
    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        num_kp,
        feature_channel,
        reshape_depth,
        compress=4,
        estimate_occlusion_map=False,
        timing=False,
    ):
        super().__init__()
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = nn.BatchNorm3d(compress, affine=True)

        self.num_kp = num_kp
        self.reshape_depth = reshape_depth
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        # ✅ 핵심: hourglass를 light 버전으로 교체
        self.hourglass = HourglassLight(
            block_expansion=block_expansion,
            in_features=(num_kp + 1) * (compress + 1),
            num_blocks=num_blocks,
            max_features=max_features,
        )

        self.mask = nn.Conv3d(
            in_channels=self.hourglass.out_filters,
            out_channels=num_kp + 1,
            kernel_size=7,
            padding=3,
        )

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(
                in_channels=self.hourglass.out_filters * reshape_depth,
                out_channels=1,
                kernel_size=7,
                padding=3,
            )

        # Timing statistics
        self.timing_stats = {
            'phase1_compress_ms': [],
            'phase2_sparse_motion_ms': [],
            'phase3_deformed_feature_ms': [],
            'phase4_heatmap_ms': [],
            'phase5_input_prep_ms': [],
            'phase6_hourglass_ms': [],
            'phase7_mask_ms': [],
            'phase8_deformation_ms': [],
            'phase9_occlusion_ms': [],
        }
        self.enable_timing = True
    
    def enable_timing_stats(self, enable=True):
        """Enable/disable timing statistics collection"""
        self.enable_timing = enable
        if enable:
            # Reset stats
            for key in self.timing_stats:
                self.timing_stats[key] = []

    def get_timing_stats(self):
        """Get timing statistics summary"""
        stats = {}
        for key, values in self.timing_stats.items():
            if values:
                stats[key] = {
                    'mean_ms': sum(values) / len(values),
                    'total_ms': sum(values),
                    'count': len(values),
                    'min_ms': min(values),
                    'max_ms': max(values),
                }
            else:
                stats[key] = {'mean_ms': 0, 'total_ms': 0, 'count': 0, 'min_ms': 0, 'max_ms': 0}
        return stats

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=feature.type()).view(1, 1, d, h, w, 3)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)

        kp_driving = kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)
        kp_source = kp_source.view(bs, self.num_kp, 1, 1, 1, 3)

        sparse_motions = identity_grid - kp_driving + kp_source
        # background(0) motion 포함
        sparse_motions = torch.cat([identity_grid, sparse_motions], dim=1)
        return sparse_motions  # (bs, 1+num_kp, d, h, w, 3)

    def create_deformed_feature(self, feature, sparse_motions):
        bs, c, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).repeat(1, sparse_motions.shape[1], 1, 1, 1, 1)
        feature_repeat = feature_repeat.view(bs * sparse_motions.shape[1], c, d, h, w)

        sparse_motions = sparse_motions.view(bs * sparse_motions.shape[1], d, h, w, 3)
        deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=True)
        deformed = deformed.view(bs, -1, c, d, h, w)
        return deformed

    def create_heatmap_representations(self, deformed_feature, kp_driving, kp_source):
        bs = deformed_feature.shape[0]
        spatial_size = deformed_feature.shape[-3:]  # (d,h,w)

        heatmap_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        heatmap_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)

        heatmap = heatmap_driving - heatmap_source
        # background(0) 채널
        zeros = torch.zeros(bs, 1, *spatial_size, device=heatmap.device, dtype=heatmap.dtype)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        return heatmap.unsqueeze(2)  # (bs, 1+num_kp, 1, d, h, w)

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # e.g. (bs,32,16,64,64)

        # Phase 1: feature compress
        feature = self.compress(feature)      # (bs, compress, d,h,w)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = {}

        # Phase 2: sparse motion
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)

        # Phase 3: deformed feature
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        # Phase 4: heatmap
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        # Phase 5: hourglass input prep
        hg_input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs,1+K, 1+compress, d,h,w)
        hg_input = hg_input.view(bs, -1, d, h, w)                 # (bs, (1+K)*(1+compress), d,h,w)

        # Phase 6: hourglass
        prediction = self.hourglass(hg_input)

        # Phase 7: mask
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs,1+K,d,h,w)
        out_dict["mask"] = mask

        # Phase 8: deformation
        mask_ = mask.unsqueeze(2)  # (bs,1+K,1,d,h,w)
        sparse_motion_ = sparse_motion.permute(0, 1, 5, 2, 3, 4)  # (bs,1+K,3,d,h,w)
        deformation = (sparse_motion_ * mask_).sum(dim=1)         # (bs,3,d,h,w)
        deformation = deformation.permute(0, 2, 3, 4, 1)          # (bs,d,h,w,3)
        out_dict["deformation"] = deformation

        # Phase 9: occlusion(optional)
        if self.flag_estimate_occlusion_map:
            bs2, ch, d2, h2, w2 = prediction.shape
            pred2d = prediction.view(bs2, ch * d2, h2, w2)
            occlusion_map = torch.sigmoid(self.occlusion(pred2d))
            out_dict["occlusion_map"] = occlusion_map

        return out_dict

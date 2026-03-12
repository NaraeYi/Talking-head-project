# import cv2
import numpy as np
# from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask

import torch
import torch.nn.functional as F

# CPU PutBack을 이 파일에 같이 두고 싶다면, OpenCV 의존성은 선택적으로 로드
try:
    import cv2  # only needed for CPU PutBack path
except Exception:
    cv2 = None

try:
    from ..utils.blend import blend_images_cy  # only needed for CPU PutBack path
except Exception:
    blend_images_cy = None


class PutBackTorchGPU:
    def __init__(self, mask_ori_float_512: np.ndarray, align_corners: bool = True, device: str = "cuda"):
        """
        mask_ori_float_512: (512,512) float32, range [0,1]
        """
        assert mask_ori_float_512.shape == (512, 512)
        self.align_corners = align_corners
        self.device = device

        # (1,1,512,512) torch GPU 템플릿로 만들어두고 재사용
        self.mask_src = torch.from_numpy(mask_ori_float_512).float()[None, None, :, :]  # CPU
        # 실제 실행 시 .to('cuda')는 forward에서 (한 번만) 처리하도록 함
        self.mask_src_gpu = None

    @staticmethod
    def _to_homography_3x3(M_2x3_or_3x3):
        M = np.asarray(M_2x3_or_3x3, dtype=np.float32)
        if M.shape == (2, 3):
            H = np.eye(3, dtype=np.float32)
            H[:2, :] = M
            return H
        assert M.shape == (3, 3)
        return M

    def _pixel_to_norm_mat(self, W, H, device):
        """
        Homogeneous 3x3 matrix that maps pixel coords -> normalized coords [-1,1]
        For align_corners=True:
          x_norm = 2*x/(W-1) - 1
          y_norm = 2*y/(H-1) - 1
        """
        if self.align_corners:
            sx = 2.0 / (W - 1)
            sy = 2.0 / (H - 1)
            tx = -1.0
            ty = -1.0
        else:
            # align_corners=False:
            # x_norm = (2*x + 1)/W - 1
            # y_norm = (2*y + 1)/H - 1
            sx = 2.0 / W
            sy = 2.0 / H
            tx = -1.0 + 1.0 / W
            ty = -1.0 + 1.0 / H

        T = torch.tensor([
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)
        return T

    def _norm_to_pixel_mat(self, W, H, device):
        """
        Homogeneous 3x3 matrix that maps normalized coords [-1,1] -> pixel coords
        Inverse of _pixel_to_norm_mat
        """
        T = self._pixel_to_norm_mat(W, H, device)
        return torch.linalg.inv(T)

    # (1) 역행렬로 방향을 바꾸고 + (2) 픽셀좌표↔정규화좌표 변환까지 포함
    # PyTorch affine_grid에 넣을 수 있는 theta를 만든 함수
    # OpenCV 𝑀𝑐2𝑜: src(crop) → dst(orig)
    # PyTorch grid_sample: dst(orig) → src(crop) 가 필요 → 그래서 inverse가 필요하고, 그걸 _opencv_M_to_theta()에서 처리
    def _opencv_M_to_theta(self, M_c2o, H_out, W_out, device):
        """
        OpenCV warpAffine uses forward mapping src->dst:
          p_dst = M_c2o * p_src   (crop -> orig)
        But torch affine_grid expects mapping from output -> input:
          p_in_norm = theta * p_out_norm
        So we invert M first to get dst->src, then wrap with normalization matrices.
        """
        # OpenCV M (crop->orig)
        H_c2o = self._to_homography_3x3(M_c2o)
        H_o2c = np.linalg.inv(H_c2o).astype(np.float32)  # orig->crop (dst->src)
        H_o2c = torch.from_numpy(H_o2c).to(device=device, dtype=torch.float32)

        # Build normalization transforms
        # out = original frame coords (W_out,H_out)
        # in  = crop coords (512,512)
        T_out_norm2pix = self._norm_to_pixel_mat(W_out, H_out, device)     # out_norm -> out_pix
        T_in_pix2norm  = self._pixel_to_norm_mat(512, 512, device)         # in_pix -> in_norm

        # Compose: out_norm -> out_pix -> in_pix -> in_norm
        # in_norm = T_in_pix2norm @ H_o2c @ T_out_norm2pix @ out_norm
        A = T_in_pix2norm @ H_o2c @ T_out_norm2pix

        theta = A[:2, :]  # 2x3
        return theta[None, :, :]  # (1,2,3)

    @torch.no_grad()
    # def __call__(self, frame_rgb_u8: np.ndarray, render_img_u8_or_f32, M_c2o):
    def __call__(self, frame_rgb_u8_or_torch, render_img_u8_or_f32, M_c2o, return_gpu_tensor: bool = True):
        """
        # frame_rgb_u8: (H,W,3) uint8 RGB
        frame_rgb_u8_or_torch:
            - CPU np.uint8 (H,W,3) RGB
            - OR torch uint8/float tensor on GPU:
              * (H,W,3) uint8 RGB
              * (1,3,H,W) float in [0,1]
        render_img_u8_or_f32:
          - (512,512,3) uint8 RGB OR
          - (1,3,512,512) torch CUDA float in [0,1] (권장: TRT out_t)
        M_c2o: (2,3) or (3,3) crop->orig
        returns: (H,W,3) uint8 RGB
        """
        # device = "cuda"
        device = self.device

        # H_out, W_out = frame_rgb_u8.shape[:2]
        # --- background to GPU float (1,3,H,W) ---
        if torch.is_tensor(frame_rgb_u8_or_torch):
            bg_in = frame_rgb_u8_or_torch
            if bg_in.dim() == 3 and bg_in.shape[-1] == 3:
                # (H,W,3) uint8
                bg = bg_in.to(device=device, non_blocking=True)
                if bg.dtype != torch.uint8:
                    bg = bg.to(torch.uint8)
                H_out, W_out = bg.shape[0], bg.shape[1]
                bg = (bg.float() / 255.0).permute(2,0,1).unsqueeze(0).contiguous()
            elif bg_in.dim() == 4 and bg_in.shape[1] == 3:
                # already (1,3,H,W) float
                bg = bg_in.to(device=device, non_blocking=True).contiguous()
                H_out, W_out = bg.shape[2], bg.shape[3]
                if bg.dtype != torch.float32:
                    bg = bg.float()
                # assume already 0~1
            else:
                raise ValueError("frame_rgb torch tensor must be (H,W,3) or (1,3,H,W)")
        else:
            frame_rgb_u8 = np.asarray(frame_rgb_u8_or_torch)
            H_out, W_out = frame_rgb_u8.shape[:2]
            bg = torch.from_numpy(frame_rgb_u8).to(device=device, dtype=torch.float32) / 255.0
            bg = bg.permute(2,0,1).unsqueeze(0).contiguous()  # (1,3,H,W)

        # # background to GPU float
        # bg = torch.from_numpy(frame_rgb_u8).to(device=device, dtype=torch.float32) / 255.0
        # bg = bg.permute(2,0,1).unsqueeze(0).contiguous()  # (1,3,H,W)

        # render image to GPU float (1,3,512,512)
        if torch.is_tensor(render_img_u8_or_f32):
            fg = render_img_u8_or_f32
            if fg.dim() == 4 and fg.shape[1] == 3:
                # assume (1,3,512,512) float
                fg = fg.to(device=device)
            else:
                raise ValueError("render_img torch tensor must be (1,3,512,512)")
        else:
            rend = np.asarray(render_img_u8_or_f32)
            assert rend.shape == (512,512,3)
            fg = torch.from_numpy(rend).to(device=device, dtype=torch.float32) / 255.0
            fg = fg.permute(2,0,1).unsqueeze(0).contiguous()

        # mask source to GPU (1,1,512,512)
        if self.mask_src_gpu is None:
            self.mask_src_gpu = self.mask_src.to(device=device, dtype=torch.float32)

        # theta for warping to original frame size
        theta = self._opencv_M_to_theta(M_c2o, H_out=H_out, W_out=W_out, device=device)  # (1,2,3)

        # build grid and warp fg/mask into original frame space
        grid = F.affine_grid(theta, size=(1, 3, H_out, W_out), align_corners=self.align_corners)
        fg_warp = F.grid_sample(fg, grid, mode="bilinear", padding_mode="zeros", align_corners=self.align_corners)
        mask_warp = F.grid_sample(self.mask_src_gpu, grid, mode="bilinear", padding_mode="zeros", align_corners=self.align_corners)
        mask_warp = mask_warp.clamp(0.0, 1.0)

        # alpha blending on GPU
        out = bg * (1.0 - mask_warp) + fg_warp * mask_warp

        # back to uint8 HWC on CPU/ 매 프레임 GPU→CPU 동기화(sync) & 느려짐 원인!!
        # out_u8 = (out[0].permute(1,2,0).clamp(0,1) * 255.0).to(torch.uint8).cpu().numpy()
        # return out_u8

        # ★ 변경: GPU uint8 텐서를 만들고 반환 옵션 제공
        out_u8_gpu = (out * 255.0).clamp(0, 255).to(torch.uint8)   # (1,3,H,W) GPU

        if return_gpu_tensor:
            return out_u8_gpu   # GPU tensor (1,3,H,W) uint8

        # (fallback) 기존처럼 CPU numpy 반환이 필요하면 아래 유지
        out_u8 = out_u8_gpu[0].permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3) uint8
        return out_u8



class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

class PutBack:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if cv2 is None:
            raise ImportError("cv2 is required for PutBack (CPU) mask_template_path loading.")

        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        if blend_images_cy is None:
            # numpy fallback (조금 느리지만 안전)
            result = mask_warped[..., None] * frame_warped + (1 - mask_warped[..., None]) * frame_rgb
            self.result_buffer[:] = np.clip(result, 0, 255).astype(np.uint8)
        else:
            blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer
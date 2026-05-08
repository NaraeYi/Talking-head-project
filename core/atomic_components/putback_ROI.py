import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackROI:
    def __init__(self, mask_template_path=None, margin=4):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:, :, 0]
        self.margin = margin
        self.result_roi_buffer = None

    def _compute_roi_from_M(self, M_c2o, src_w, src_h, dst_w, dst_h):
        corners = np.array([
            [0, 0],
            [src_w - 1, 0],
            [src_w - 1, src_h - 1],
            [0, src_h - 1],
        ], dtype=np.float32).reshape(-1, 1, 2)

        warped_corners = cv2.transform(corners, M_c2o[:2, :]).reshape(-1, 2)

        x_coords = warped_corners[:, 0]
        y_coords = warped_corners[:, 1]

        x_min = max(0, int(np.floor(x_coords.min())) - self.margin)
        y_min = max(0, int(np.floor(y_coords.min())) - self.margin)
        x_max = min(dst_w, int(np.ceil(x_coords.max())) + self.margin)
        y_max = min(dst_h, int(np.ceil(y_coords.max())) + self.margin)

        if x_max <= x_min or y_max <= y_min:
            return None

        return x_min, y_min, x_max, y_max

    def _shift_M_for_roi(self, M_c2o, x_min, y_min):
        M_roi = M_c2o[:2, :].copy()
        M_roi[0, 2] -= x_min
        M_roi[1, 2] -= y_min
        return M_roi

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        src_h, src_w = render_image.shape[:2]

        roi = self._compute_roi_from_M(M_c2o, src_w, src_h, w, h)
        if roi is None:
            return frame_rgb

        x_min, y_min, x_max, y_max = roi
        roi_w = x_max - x_min
        roi_h = y_max - y_min

        M_roi = self._shift_M_for_roi(M_c2o, x_min, y_min)

        mask_warped = cv2.warpAffine(
            self.mask_ori_float,
            M_roi,
            dsize=(roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).clip(0, 1)

        frame_warped = cv2.warpAffine(
            render_image,
            M_roi,
            dsize=(roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        frame_roi = frame_rgb[y_min:y_max, x_min:x_max]

        if self.result_roi_buffer is None or self.result_roi_buffer.shape != (roi_h, roi_w, 3):
            self.result_roi_buffer = np.empty((roi_h, roi_w, 3), dtype=np.uint8)

        blend_images_cy(mask_warped, frame_warped, frame_roi, self.result_roi_buffer)

        # result = frame_rgb.copy()
        # result[y_min:y_max, x_min:x_max] = self.result_roi_buffer
        # return result
        frame_rgb[y_min:y_max, x_min:x_max] = self.result_roi_buffer
        return frame_rgb

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
        # frame_rgb = np.ascontiguousarray(frame_rgb)
        # render_image = np.ascontiguousarray(render_image)

        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            # self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_NEAREST
        ).clip(0, 1)
        # mask_warped = cv2.warpAffine(
        #     self.mask_ori_float,
        #     M_c2o[:2, :],
        #     dsize=(w, h),
        #     flags=cv2.INTER_NEAREST,            # ★ 마스크만 nearest
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=0.0,
        # )
        # # clip은 유지해도 되지만, borderValue=0이면 보통 필요가 줄어듦
        # mask_warped = mask_warped.clip(0, 1)

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
        # if self.result_buffer is None or self.result_buffer.shape != (h, w, 3):
        #     self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer
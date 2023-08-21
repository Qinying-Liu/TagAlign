# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder
from utils import get_logger


class TCLSegInference(EncoderDecoder):
    def __init__(
        self,
        model,
        text_embedding,
        with_bg,
        test_cfg=dict(),
        pamr=False,
        bg_thresh=0.5,
        kp_w=0.3,
        **kwargs,
    ):
        super(EncoderDecoder, self).__init__()  # init BaseSegmenter (parent of EncoderDecoder)

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.kp_w = kp_w

        self.model = model
        self.model.clip_image_encoder.clip_visual.conv1.stride = (4, 4)
        self.register_buffer("text_embedding", text_embedding)
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building TCLSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}, kp_w={kp_w}"
        )

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        masks, simmap = self.model.generate_masks(
            img,
            self.text_embedding,
            apply_pamr=self.pamr,
            kp_w=self.kp_w,
        )

        B, N, H, W = masks.shape

        if self.with_bg:
            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)

        return masks

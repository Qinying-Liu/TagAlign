import mmcv
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder


class TagAlignSegInference(EncoderDecoder):
    def __init__(
        self,
        model,
        text_embedding,
        with_bg,
        test_cfg=dict(),
        bg_thresh=0.5,
        clip_w=0.3,
        scale=10, 
        bias=-2.5,
        **kwargs,
    ):
        super(EncoderDecoder, self).__init__()  # init BaseSegmenter (parent of EncoderDecoder)

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.bg_thresh = bg_thresh
        self.clip_w = clip_w

        self.scale = scale
        self.bias = bias

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        if self.with_bg:
            self.out_channels = len(text_embedding) + 1
        else:
            self.out_channels = len(text_embedding)

        self.align_corners = False

    def encode_decode(self, img, img_meta):
        masks = self.model.generate_masks(
            img,
            self.text_embedding,
            clip_w=self.clip_w,
            scale=self.scale,
            bias=self.bias
        )

        B, N, H, W = masks.shape
        if self.with_bg:
            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
        return masks

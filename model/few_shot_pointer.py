import os

import torch
import torch.nn.functional as F
from PIL import Image

from model.vision_backbone.dinov2_vit import Dinov2Module
from model.vision_backbone.dinov3_vit import Dinov3Module

from model.vlm_modules.qwen2_5_vl.qwen2_5_vl import Qwen2_5_VLModule
from model.vlm_modules.molmo.molmo import MolmoModule
from model.vlm_modules.internvl3.internvl3 import InternVL3_VLModule
from model.vlm_modules.ovis.ovis import Ovis2_5_VLModule
from model.vlm_modules.kimi_vl.kimi_vl import Kimi_VLModule


# Keyed by a substring that must appear (case-insensitive) in the checkpoint name.
_VLM_REGISTRY = {
    'qwen':      Qwen2_5_VLModule,
    'molmo':     MolmoModule,
    'internvl3': InternVL3_VLModule,
    'ovis':      Ovis2_5_VLModule,
    'kimi':      Kimi_VLModule,
}

_BACKBONE_REGISTRY = {
    'dinov2': Dinov2Module,
    'dinov3': Dinov3Module,
}

_SELECTED_HEADS_FILES = {
    'qwen':      'qwen2_5_vl_7b.json',
    'molmo':     'molmo_d_7b.json',
    'internvl3': 'internvl_3_8b.json',
    'ovis':      'ovis2_5_9b.json',
    'kimi':      'kimi_vl_a3b.json',
}

_SELECTED_HEADS_DIR = os.path.join(
    os.path.dirname(__file__), 'vlm_modules', 'vlm_selected_heads_files'
)


def _select_from_registry(registry, checkpoint, kind):
    checkpoint_lower = checkpoint.lower()
    matches = [cls for key, cls in registry.items() if key in checkpoint_lower]
    if not matches:
        raise ValueError(
            f"No {kind} matches checkpoint {checkpoint!r}. "
            f"Expected one of: {list(registry)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {kind} checkpoint {checkpoint!r} matched multiple entries."
        )
    return matches[0]


def _infer_selected_heads_file(vlm_checkpoint):
    checkpoint_lower = vlm_checkpoint.lower()
    for key, filename in _SELECTED_HEADS_FILES.items():
        if key in checkpoint_lower:
            return os.path.join(_SELECTED_HEADS_DIR, filename)
    raise ValueError(
        f"Cannot infer selected-heads file for checkpoint {vlm_checkpoint!r}."
    )


class FewShotPointer():
    def __init__(self, config):
        self.config = config

        if getattr(config, 'selected_heads_file', None) is None:
            config.selected_heads_file = _infer_selected_heads_file(config.vlm_checkpoint)

        vlm_cls = _select_from_registry(_VLM_REGISTRY, config.vlm_checkpoint, 'VLM')
        print(f"Loading {vlm_cls.__name__}")
        self.vlm_module = vlm_cls(config)

        backbone_cls = _select_from_registry(
            _BACKBONE_REGISTRY, config.vision_backbone_checkpoint, 'vision backbone'
        )
        print(f"Loading {backbone_cls.__name__}")
        self.vision_backbone = backbone_cls(config)

    def center_point(
        self,
        score_map,
        patch_size,
        ori_size,
        input_size
    ):
        ori_height, ori_width = ori_size
        input_height, input_width = input_size
        flat_idx = torch.argmax(score_map)
        max_index = torch.unravel_index(flat_idx, score_map.shape)
        y = int((max_index[0] * patch_size + patch_size // 2) * ori_height / input_height)
        x = int((max_index[1] * patch_size + patch_size // 2) * ori_width / input_width)
        return y, x

    def k_shot_pointing(self,
        k,
        target_image_file,
        target_query,
        reference_image_file_list,
        reference_point_list,
    ):
        assert k == len(reference_image_file_list)
        assert k == len(reference_point_list)

        # Obtain text-to-image score
        text_to_image_score = self.vlm_module.get_text_to_image_score(target_image_file, target_query)

        # Obtain image-to-image score
        image_to_image_score = self.vision_backbone.get_image_to_image_score(
            target_image_file,
            reference_image_file_list,
            reference_point_list,
        )

        # Fuse text-to-image and image-to-image
        H = W = self.vision_backbone.image_size // self.vision_backbone.patch_size

        text_to_image_score = text_to_image_score.unsqueeze(0).unsqueeze(0)
        text_to_image_score = F.interpolate(text_to_image_score, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        fused_score = (image_to_image_score * text_to_image_score).unsqueeze(0).unsqueeze(0)
        final_score = F.interpolate(fused_score, size=(2*H, 2*W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Take center point of patch as prediction
        ori_img = Image.open(target_image_file).convert("RGB")
        ori_width, ori_height = ori_img.size
        pred_point = self.center_point(final_score, self.vision_backbone.patch_size // 2, (ori_height, ori_width), (self.vision_backbone.image_size, self.vision_backbone.image_size))

        return pred_point

from model.vision_backbone.dino_base import DinoModule


class Dinov3Module(DinoModule):
    image_size = 1024
    patch_size = 16
    num_prefix_tokens = 5  # [CLS] + 4 register tokens

from model.vision_backbone.dino_base import DinoModule


class Dinov2Module(DinoModule):
    image_size = 896
    patch_size = 14
    num_prefix_tokens = 1  # [CLS]

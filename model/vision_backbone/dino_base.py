import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoModule:
    """Shared base for DINOv2 / DINOv3 feature extraction and visual correspondence."""

    image_size: int = None
    patch_size: int = None
    # Number of leading tokens in last_hidden_state that are NOT spatial patches
    # (e.g. 1 for [CLS] in DINOv2, 5 for [CLS] + 4 register tokens in DINOv3).
    num_prefix_tokens: int = None

    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.load_model()

    def load_model(self):
        self.processor = AutoImageProcessor.from_pretrained(
            self.config.vision_backbone_checkpoint,
            device_map=self.config.device,
        )
        self.processor.size = {"height": self.image_size, "width": self.image_size}
        self.processor.do_center_crop = False
        self.processor.crop_size = self.image_size
        self.processor.do_resize = True

        self.model = AutoModel.from_pretrained(
            self.config.vision_backbone_checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
        ).eval()

    def _encode(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device=self.config.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, self.num_prefix_tokens:, :]

    @torch.no_grad()
    def get_reference_features(self, reference_image_file, reference_point):
        if isinstance(reference_image_file, str):
            reference_img = Image.open(reference_image_file).convert("RGB")
        else:
            reference_img = reference_image_file
        reference_width, reference_height = reference_img.size

        reference_last_hidden_state = self._encode(reference_img)

        # Map the annotated point to the patch containing it.
        y, x = reference_point
        x_patch = int(x / reference_width * self.image_size) // self.patch_size
        y_patch = int(y / reference_height * self.image_size) // self.patch_size
        reference_single_patch_feature = reference_last_hidden_state[
            int(y_patch * (self.image_size / self.patch_size) + x_patch)
        ]
        return reference_last_hidden_state, reference_single_patch_feature

    @torch.no_grad()
    def compute_backward_score(
        self,
        target_last_hidden_state,
        reference_last_hidden_state,
        single_patch_feature,
    ):
        target_norm = F.normalize(target_last_hidden_state, dim=1)
        reference_norm = F.normalize(reference_last_hidden_state, dim=1)

        similarity = target_norm @ reference_norm.T
        best_idxs = similarity.argmax(dim=1)

        reference_matched = reference_norm[best_idxs]
        if single_patch_feature.dim() == 1:
            single_patch_feature = single_patch_feature.unsqueeze(0)
        single_patch_feature_norm = F.normalize(single_patch_feature, dim=1)

        return reference_matched @ single_patch_feature_norm.T

    @torch.no_grad()
    def get_image_to_image_score(
        self,
        target_image_file,
        reference_image_file_list,
        reference_point_list,
    ):
        if isinstance(target_image_file, str):
            target_img = Image.open(target_image_file).convert("RGB")
        else:
            target_img = target_image_file

        target_last_hidden_state = self._encode(target_img)

        grid = self.image_size // self.patch_size
        image_to_image_score = torch.ones((grid, grid), device=self.config.device)

        for reference_image_file, reference_point in zip(reference_image_file_list, reference_point_list):
            reference_last_hidden_state, reference_single_patch_feature = self.get_reference_features(
                reference_image_file, reference_point
            )

            # Forward score: target patches vs. the support patch containing the labelled point.
            # ``reference_single_patch_feature`` is 1D [d], so the matmul reduces to a dot product
            # per target patch (no transpose needed).
            forward_score = (
                F.normalize(target_last_hidden_state, dim=1)
                @ F.normalize(reference_single_patch_feature, dim=0)
            ).reshape(grid, grid)

            # Backward score: each target patch -> its nearest support patch -> similarity to labelled patch.
            backward_score = self.compute_backward_score(
                target_last_hidden_state,
                reference_last_hidden_state,
                reference_single_patch_feature,
            ).reshape(grid, grid)

            combined = forward_score * backward_score
            combined = (combined - combined.min()) / (combined.max() - combined.min())
            image_to_image_score *= combined

        return image_to_image_score

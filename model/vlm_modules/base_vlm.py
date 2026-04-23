import json

import torch
from torchvision.transforms.functional import gaussian_blur


class BaseVLMModule:
    """Shared scaffolding for VLM wrappers.

    Subclasses implement :py:meth:`load_model` and :py:meth:`get_text_to_image_score`,
    and may use the helpers here for selected-head loading and attention aggregation.
    """

    question_template = '{query}'

    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.selected_heads_list = None
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def get_text_to_image_score(self, image_file, query):
        raise NotImplementedError

    def _load_selected_heads(self, key=None):
        """Parse the selected-heads JSON into a list of ``(layer, head)`` tuples.

        Args:
            key: If given, reads ``data[key]`` before truncation (used by Molmo, whose JSON
                has separate ``'high_resolution'`` / ``'low_resolution'`` sections).
        """
        with open(self.config.selected_heads_file) as f:
            data = json.load(f)
        if key is not None:
            data = data[key]
        entries = data[:self.config.num_selected_heads]
        return [(e['layer'], e['head']) for e in entries]

    @staticmethod
    def aggregate_selected_heads(attentions, selected_heads_list):
        """Reduce per-head attention maps to a single min-max normalized map.

        Args:
            attentions: Tensor of shape ``[num_layers, num_heads, H, W]``.
            selected_heads_list: List of ``(layer, head)`` tuples. If ``None``, sums over
                all layers and heads instead.

        Returns:
            Tensor of shape ``[H, W]`` normalized to ``[0, 1]``.
        """
        _, _, h, w = attentions.shape
        acc = torch.zeros((h, w), device=attentions.device)
        if selected_heads_list is not None:
            for (layer, head) in selected_heads_list:
                acc += gaussian_blur(
                    attentions[layer, head].unsqueeze(0),
                    kernel_size=[7, 7],
                    sigma=1.0,
                ).squeeze(0)
        else:
            acc = attentions.sum(dim=(0, 1))
        return (acc - acc.min()) / (acc.max() - acc.min())

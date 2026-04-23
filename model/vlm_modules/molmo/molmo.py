import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from model.vlm_modules.base_vlm import BaseVLMModule
from model.vlm_modules.molmo.molmo_processor import MolmoProcessor
from model.vlm_modules.molmo.image_preprocessing_molmo import MolmoImageProcessor
from model.vlm_modules.molmo.modeling_molmo import MolmoForCausalLM

special_token_dict = {
    "ĠAssistant": 21388,
    "<im_col>":   152067,
    "<im_end>":   152065,
    "<im_patch>": 152066,
    "<im_start>": 152064,
}


class MolmoModule(BaseVLMModule):
    """Molmo uses dual-resolution inputs (thumbnail + high-res crops) and maintains
    separate selected-head lists for each stream, which is why its aggregation logic
    does not share the :py:meth:`aggregate_selected_heads` helper."""

    def __init__(self, config):
        self.high_res_selected_heads_list = None
        self.low_res_selected_heads_list = None
        super().__init__(config)

    def load_model(self):
        self.processor = MolmoProcessor.from_pretrained(
            self.config.vlm_checkpoint, trust_remote_code=True, device_map=self.config.device
        )
        self.processor.image_processor = MolmoImageProcessor(
            max_crops=12,
            overlap_margins=(4, 4),
            base_image_input_size=(336, 336),
            image_token_length_w=12,
            image_token_length_h=12,
            image_patch_size=14,
            image_padding_mask=True,
            do_normalize=True,
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.model = MolmoForCausalLM.from_pretrained(
            self.config.vlm_checkpoint, device_map=self.config.device, trust_remote_code=True
        )
        self.high_res_selected_heads_list = self._load_selected_heads(key='high_resolution')
        self.low_res_selected_heads_list = self._load_selected_heads(key='low_resolution')

    def prepare_inputs(self, image_file, query):
        question = self.question_template.format(query=query)
        inputs = self.processor.process(images=[Image.open(image_file)], text=question)
        return {k: v.to(self.config.device).unsqueeze(0) for k, v in inputs.items()}

    @staticmethod
    def _accumulate_stream(accumulator, attentions, heads_list):
        """Molmo's per-stream aggregation: normalize each head's blurred map to [0, 1]
        before adding to the accumulator. Falls back to a raw sum if no heads are given."""
        if heads_list is not None:
            for (layer, head) in heads_list:
                score = gaussian_blur(
                    attentions[layer, head].unsqueeze(0), kernel_size=[7, 7], sigma=1.0
                ).squeeze(0)
                score = (score - score.min()) / (score.max() - score.min())
                accumulator += score
        else:
            accumulator += attentions.sum(dim=(0, 1))
        return accumulator

    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        inputs = self.prepare_inputs(image_file, query)

        attentions = self.model.get_attentions(**inputs)
        attentions = torch.cat(attentions, 0)

        image_start_pos = (inputs['input_ids'][0] == special_token_dict['<im_start>']).nonzero(as_tuple=True)[0] + 1
        image_end_pos = (inputs['input_ids'][0] == special_token_dict['<im_end>']).nonzero(as_tuple=True)[0]
        assistant_id_pos = (inputs['input_ids'][0] == special_token_dict['ĠAssistant']).nonzero(as_tuple=True)[0]

        image_patch_pos = []
        num_cols = []
        for img_start, img_end in zip(image_start_pos, image_end_pos):
            image_patch_pos.append(inputs['input_ids'][0, img_start:img_end] == special_token_dict['<im_patch>'])
            num_cols.append(sum(inputs['input_ids'][0, img_start:img_end] == special_token_dict['<im_col>']))

        # Stream 0 is the low-res thumbnail; stream 1 is the high-res crops.
        nail_thumb_attention = attentions[:, :, assistant_id_pos - 1, image_start_pos[0]:image_end_pos[0]][:, :, 0, image_patch_pos[0]]
        high_res_attention = attentions[:, :, assistant_id_pos - 1, image_start_pos[1]:image_end_pos[1]][:, :, 0, image_patch_pos[1]]

        num_layers, num_heads, _ = high_res_attention.shape
        nail_thumb_attention = nail_thumb_attention.reshape((num_layers, num_heads, num_cols[0], -1))
        high_res_attention = high_res_attention.reshape((num_layers, num_heads, num_cols[1], -1))

        _, _, target_h, target_w = high_res_attention.shape
        nail_thumb_attention = F.interpolate(
            nail_thumb_attention, size=(target_h, target_w), mode='bilinear', align_corners=False
        )

        text_to_image_attentions = torch.ones((target_h, target_w), device=self.config.device)
        text_to_image_attentions = self._accumulate_stream(
            text_to_image_attentions, nail_thumb_attention, self.low_res_selected_heads_list
        )
        text_to_image_attentions = self._accumulate_stream(
            text_to_image_attentions, high_res_attention, self.high_res_selected_heads_list
        )

        return (text_to_image_attentions - text_to_image_attentions.min()) / (
            text_to_image_attentions.max() - text_to_image_attentions.min()
        )

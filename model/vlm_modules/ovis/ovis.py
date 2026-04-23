import torch

from PIL import Image
from transformers import AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

from model.vlm_modules.base_vlm import BaseVLMModule

special_token_dict = {
    '<|vision_start|>': 151652,
    '<|vision_end|>':   151653,
    '<|image_pad|>':    151655,
    '<|vision_pad|>':   151654,
    '<|im_end|>':       151645,
    '</':               522,
}

# Ovis marks image token positions with -300 in the input_ids.
OVIS_IMAGE_TOKEN_ID = -300


class Ovis2_5_VLModule(BaseVLMModule):
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.vlm_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            trust_remote_code=True,
            attn_implementation='eager',
        )
        self.selected_heads_list = self._load_selected_heads()

    def get_image_input_size(self, img_path):
        msg = [{"role": "user", "content": [{"type": "image", "image": img_path}]}]
        imgs, vids = process_vision_info(msg)
        txt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[txt], images=imgs, videos=vids, padding=True, return_tensors="pt")
        return inp['image_grid_thw'][0][1] * 14, inp['image_grid_thw'][0][2] * 14

    def prepare_inputs(self, image_file, query):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_file)},
                {"type": "text", "text": self.question_template.format(query=query)},
            ],
        }]
        return self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=False,
            enable_thinking=False,
            max_pixels=1024 * 1024,
        )

    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        while query[-1] == " ":
            query = query[:-1]

        input_ids, pixel_values, grid_thws = self.prepare_inputs(image_file, query)
        query_input_ids = torch.tensor(self.model._tokenize_with_visual_placeholder(query))

        query_start_pos = -1
        for i in range(len(input_ids[0]) - len(query_input_ids) + 1):
            if torch.equal(input_ids[0][i:i + len(query_input_ids)], query_input_ids):
                query_start_pos = i
                break
        assert query_start_pos != -1

        img_pos = input_ids[0] == OVIS_IMAGE_TOKEN_ID

        results = self.model(
            input_ids=input_ids.to(self.config.device),
            pixel_values=pixel_values.to(self.config.device),
            grid_thws=grid_thws.to(self.config.device),
            attention_mask=None,
            output_attentions=True,
        )

        attentions = torch.cat(
            [results['attentions'][i].to(self.config.device) for i in range(len(results['attentions']))],
            0,
        )  # [Layer, Head, Query, Key]
        attentions = attentions[:, :, query_start_pos + len(query_input_ids) - 1, img_pos]
        attentions = attentions.reshape(*attentions.shape[:2], grid_thws[0][1] // 2, grid_thws[0][2] // 2)
        return self.aggregate_selected_heads(attentions, self.selected_heads_list)

    def get_patch_size(self):
        return 28

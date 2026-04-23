import torch

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from model.vlm_modules.base_vlm import BaseVLMModule

# Kimi-VL uses a single token ID for image placeholders in the sequence.
KIMI_IMAGE_TOKEN_ID = 163605


class Kimi_VLModule(BaseVLMModule):
    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.config.vlm_checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.vlm_checkpoint,
            device_map=self.config.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation='eager',
        )
        self.selected_heads_list = self._load_selected_heads()

    def prepare_inputs(self, image_file, query):
        image = Image.open(image_file)
        if image.size[0] <= 196 or image.size[1] <= 196:
            shorter_side = min(image.size[0], image.size[1])
            ratio = 196 / shorter_side
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.BICUBIC)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_file},
                {"type": "text", "text": query},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")
        return self.processor(
            images=image, text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.config.device)

    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        inputs = self.prepare_inputs(image_file, query)
        img_pos = torch.where(inputs['input_ids'][0] == KIMI_IMAGE_TOKEN_ID)[0]

        outputs = self.model(**inputs, output_attentions=True, use_cache=False)

        attentions = torch.cat(
            [outputs['attentions'][i].to(self.config.device) for i in range(len(outputs['attentions']))],
            0,
        )
        num_layers, num_heads = attentions.shape[:2]
        grid_h = inputs['image_grid_hws'][0][0] // 2
        grid_w = inputs['image_grid_hws'][0][1] // 2
        attentions = attentions[:, :, -2, img_pos].reshape(num_layers, num_heads, grid_h, grid_w)
        return self.aggregate_selected_heads(attentions, self.selected_heads_list)

    def get_patch_size(self):
        return 28

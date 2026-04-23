import torch

from PIL import Image

from model.vlm_modules.base_vlm import BaseVLMModule
from model.vlm_modules.internvl3.internvl.model import load_model_and_tokenizer
from model.vlm_modules.internvl3.internvl.train.dataset import build_transform
from model.vlm_modules.internvl3.internvl.conversation import get_conv_template

# Vocab token IDs for InternVL-3's image/reference tokens.
IMG_START_TOKEN_ID = 151665
IMG_END_TOKEN_ID = 151666
REF_END_TOKEN_ID = 151671

# Attention map for InternVL-3 is over the 16x16 patch grid produced by the vision tower.
_GRID_SIZE = 16


class InternVL3_VLModule(BaseVLMModule):
    def load_model(self):
        self.model, self.tokenizer = load_model_and_tokenizer(self.config)
        self.model.to(self.config.device)

        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        self.transform = build_transform(is_train=False, input_size=image_size)

        self.selected_heads_list = self._load_selected_heads()

    def prepare_inputs(self, image_file, query,
                       IMG_START_TOKEN='<img>',
                       IMG_END_TOKEN='</img>',
                       IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        question = '<image>\n' + self.question_template.format(query=query)
        image = Image.open(image_file).convert('RGB')
        pixel_values = torch.stack([self.transform(image)])

        num_patches_list = [pixel_values.shape[0]]
        assert len(pixel_values) == sum(num_patches_list)

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        template = get_conv_template(self.model.template)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            prompt = prompt.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(prompt, return_tensors='pt')
        return model_inputs['input_ids'], model_inputs['attention_mask'], pixel_values

    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        input_ids, attention_mask, pixel_values = self.prepare_inputs(image_file, query)

        image_start_pos = torch.where(input_ids == IMG_START_TOKEN_ID)[1] + 1
        image_end_pos = torch.where(input_ids == IMG_END_TOKEN_ID)[1]
        query_end_pos = torch.where(input_ids == REF_END_TOKEN_ID)[1] - 1

        results = self.model(
            pixel_values=pixel_values.to(self.config.device, torch.bfloat16),
            input_ids=input_ids.to(self.config.device),
            attention_mask=attention_mask.to(self.config.device),
            output_attentions=True,
        )

        attentions = torch.cat(
            [results['attentions'][i].to(self.config.device) for i in range(len(results['attentions']))],
            0,
        )  # [Layer, Head, Query, Key]
        attentions = attentions[:, :, query_end_pos, image_start_pos:image_end_pos]
        attentions = attentions.reshape(*attentions.shape[:2], _GRID_SIZE, _GRID_SIZE)
        return self.aggregate_selected_heads(attentions, self.selected_heads_list)

    def get_patch_size(self):
        return 28

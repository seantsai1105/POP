import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm

from torchvision.transforms.functional import gaussian_blur

try: 
    from model.vlm_modules.internvl3.internvl.model import load_model_and_tokenizer
    from model.vlm_modules.internvl3.internvl.train.dataset import build_transform, dynamic_preprocess
    from model.vlm_modules.internvl3.internvl.conversation import get_conv_template
except:
    from internvl.model import load_model_and_tokenizer
    from internvl.train.dataset import build_transform, dynamic_preprocess
    from internvl.conversation import get_conv_template
    

class InternVL3_VLModule():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.selected_heads_list = None
        self.question_template = '{query}' # 'Locate the region this sentence describes: <ref>{query}</ref>'
        self.img_start_token = 151665
        self.img_end_token = 151666
        self.ref_end_token = 151671
        self.load_model()
    
    
    def load_model(self):
        if self.processor == None or self.model == None:
            self.model, self.tokenizer = load_model_and_tokenizer(self.config)
            self.model.to(self.config.device)
            
            image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
            self.transform = build_transform(is_train=False, input_size=image_size)
            
        if self.selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                selected_heads_list = json.load(f)[:self.config.num_selected_heads]
            self.selected_heads_list = [(selected_heads['layer'], selected_heads['head']) for selected_heads in selected_heads_list]
    
    
    def prepare_inputs(self, image_file, query, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        question = '<image>\n' + self.question_template.format(query=query)
        image = Image.open(image_file).convert('RGB')
        pixel_values = [self.transform(image)]
        pixel_values = torch.stack(pixel_values)
        
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        
        template = get_conv_template(self.model.template)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        
        return input_ids, attention_mask, pixel_values
    
    
    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):        
        # Prepare Input
        input_ids, attention_mask, pixel_values = self.prepare_inputs(image_file, query)
        
        # Obtain Image Start Pos
        image_start_pos = torch.where(input_ids==self.img_start_token)[1] + 1
        image_end_pos = torch.where(input_ids==self.img_end_token)[1]

        # Obtain Query Start Pos
        query_end_pos = torch.where(input_ids==self.ref_end_token)[1] - 1

        
        # Prefiliing
        with torch.no_grad():
            results = self.model(
                pixel_values = pixel_values.to(self.config.device, torch.bfloat16),
                input_ids = input_ids.to(self.config.device),
                attention_mask=attention_mask.to(self.config.device),
                output_attentions=True
            )

        attentions = torch.cat([results['attentions'][i].to(self.config.device) for i in range(len(results['attentions']))] ,0) # [Layer, Head, Query, Key]
        attentions = attentions[:, :, query_end_pos, image_start_pos:image_end_pos]
    
        attentions = attentions.reshape(*attentions.shape[:2], 16, 16)
        text_to_image_attentions = torch.zeros((16, 16), device=self.config.device)
        
        if self.selected_heads_list != None:
            for (l, h) in self.selected_heads_list:
                text_to_image_attentions += gaussian_blur(attentions[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)
        else:
            text_to_image_attentions = attentions.sum(dim=(0, 1))
        
        text_to_image_attentions = (text_to_image_attentions - text_to_image_attentions.min()) / (text_to_image_attentions.max() - text_to_image_attentions.min())
        return text_to_image_attentions
    
    def get_patch_size(self):
        return 28

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='OpenGVLab/InternVL3-8B')
    parser.add_argument('--selected-heads-file', type=str, default='./model/vlm_modules/vlm_selected_heads_files/internvl_3_8b.json')
    parser.add_argument('--num-selected-heads', type=int, default=3)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', default=True, action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    
    model = InternVL3_VLModule(args)
    text_to_image_attentions = model.get_text_to_image_score('/home/sean/PartPointing/visualization/ori_img.png', "the bottom of the red bottle")
    
    text_to_image_attentions = text_to_image_attentions.unsqueeze(0).unsqueeze(0)
    # text_to_image_attentions = F.interpolate(text_to_image_attentions, size=(74, ), mode='bilinear', align_corners=True)# .squeeze(0).squeeze(0)
    text_to_image_attentions = F.interpolate(text_to_image_attentions, size=(336, 336)).squeeze(0).squeeze(0).cpu().numpy()
    
    
    image_path = "/home/sean/PartPointing/visualization/ori_img.png"
    image = Image.open(image_path)
    resized_image = image.resize((336, 336), Image.BICUBIC)
    img_np = np.array(resized_image)
    
    plt.imshow(img_np)
    plt.imshow(text_to_image_attentions, cmap='viridis', alpha=0.65)
    plt.axis('off')
    plt.tight_layout(pad=0)        
    plt.savefig("image_2_image.png")
    # model, tokenizer = load_model_and_tokenizer(args)
    # image_size = model.config.force_image_size or model.config.vision_config.image_size
    # use_thumbnail = model.config.use_thumbnail
    # transform = build_transform(is_train=False, input_size=image_size)
    # prompt = 'Locate the region this sentence describes: <ref>{query}</ref>'
    
    # image = '/home/sean/PartPointing/visualization/ori_img.png' # (612, 612)
    # # image = '/home/sean/PartPointing/visualization/exp_img.png' # (640, 427)
    # image = Image.open(image).convert('RGB')
    # pixel_values = [transform(image)]
    # pixel_values = torch.stack(pixel_values)
    
    # generation_config = dict(
    #     num_beams=args.num_beams,
    #     max_new_tokens=100,
    #     min_new_tokens=1,
    #     do_sample=True if args.temperature > 0 else False,
    #     temperature=args.temperature,
    # )
    # pred = model.chat(
    #     tokenizer=tokenizer,
    #     pixel_values=pixel_values,
    #     question=prompt.format(query = "the red bottle"),
    #     generation_config=generation_config,
    #     verbose=True
    # )
    # answers = [pred]
        
    # print(image_size)
    # print(use_thumbnail)
    # print(pixel_values.shape)

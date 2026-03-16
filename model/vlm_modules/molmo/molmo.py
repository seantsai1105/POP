import os
import ast
import json
import torch
import random
import argparse
import requests

import xml.etree.ElementTree as ET
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pycocotools import mask as maskUtils
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

try: 
    from model.vlm_modules.molmo.molmo_processor import MolmoProcessor
    from model.vlm_modules.molmo.image_preprocessing_molmo import MolmoImageProcessor
    from model.vlm_modules.molmo.modeling_molmo import MolmoForCausalLM
except:
    from molmo_processor import MolmoProcessor
    from image_preprocessing_molmo import MolmoImageProcessor
    from modeling_molmo import MolmoForCausalLM

special_token_dict = {
    "ĠAssistant": 21388,
    "<im_col>": 152067,
    "<im_end>": 152065,
    "<im_patch>": 152066,
    "<im_start>": 152064,
}


class MolmoModule():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.high_res_selected_heads_list = None
        self.low_res_selected_heads_list = None
        # self.question_template = 'Point to {query}'
        self.question_template = '{query}'
        self.load_model()
    
    def load_model(self):
        if self.processor == None:
            self.processor = MolmoProcessor.from_pretrained(self.config.vlm_checkpoint, trust_remote_code=True, device_map=self.config.device)
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
        if self.model == None:
            self.model = MolmoForCausalLM.from_pretrained(self.config.vlm_checkpoint, device_map=self.config.device, trust_remote_code=True)
        
        if self.high_res_selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                high_res_selected_heads_list = json.load(f)['high_resolution'][:self.config.num_selected_heads]
            self.high_res_selected_heads_list  = [(selected_heads['layer'], selected_heads['head']) for selected_heads in high_res_selected_heads_list]
            print(self.high_res_selected_heads_list)
        if self.low_res_selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                low_res_selected_heads_list = json.load(f)['low_resolution'][:self.config.num_selected_heads]
            self.low_res_selected_heads_list  = [(selected_heads['layer'], selected_heads['head']) for selected_heads in low_res_selected_heads_list]
            print(self.low_res_selected_heads_list)
            
    def prepare_inputs(self, image_file, query):
        question = self.question_template.format(query=query)
        inputs = self.processor.process(images=[Image.open(image_file)], text=question)   
        inputs = {k: v.to(self.config.device).unsqueeze(0) for k, v in inputs.items()}
        return inputs
    
    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        # Prepare Input
        inputs = self.prepare_inputs(image_file, query)
        
        with torch.no_grad():
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
        
        nail_thumb_attention = attentions[:, :, assistant_id_pos-1, image_start_pos[0]:image_end_pos[0]][:, :, 0, image_patch_pos[0]]
        high_res_attention = attentions[:, :, assistant_id_pos-1, image_start_pos[1]:image_end_pos[1]][:, :, 0, image_patch_pos[1]]
        
        l, h, _ = high_res_attention.shape
        nail_thumb_attention = nail_thumb_attention.reshape((l, h, num_cols[0], -1))
        high_res_attention = high_res_attention.reshape((l, h, num_cols[1], -1))
        
        l, h, _h, _w = high_res_attention.shape
        nail_thumb_attention = F.interpolate(nail_thumb_attention, size=(_h, _w), mode='bilinear', align_corners=False)
        text_to_image_attentions = torch.ones((_h, _w), device=self.config.device)
        
        if self.low_res_selected_heads_list != None:
            for (l, h) in self.low_res_selected_heads_list:
                low_res_score = gaussian_blur(nail_thumb_attention[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)
                low_res_score = (low_res_score - low_res_score.min()) / (low_res_score.max() - low_res_score.min())
                text_to_image_attentions +=low_res_score
        else:
            text_to_image_attentions += nail_thumb_attention.sum(dim=(0, 1))
        
        if self.high_res_selected_heads_list != None:
            for (l, h) in self.high_res_selected_heads_list:
                high_res_score = gaussian_blur(high_res_attention[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)
                high_res_score = (high_res_score - high_res_score.min()) / (high_res_score.max() - high_res_score.min())
                text_to_image_attentions += high_res_score
        else:
            text_to_image_attentions += high_res_attention.sum(dim=(0, 1))
        # text_to_image_attentions += nail_thumb_attention.amax(dim=(0, 1))
        # text_to_image_attentions += high_res_attention.amax(dim=(0, 1))
        
        text_to_image_attentions = (text_to_image_attentions - text_to_image_attentions.min()) / (text_to_image_attentions.max() - text_to_image_attentions.min())
        return text_to_image_attentions
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm-checkpoint', type=str, default='allenai/Molmo-7B-D-0924')
    parser.add_argument('--selected-heads-file', type=str, default='./model/vlm_modules/vlm_selected_heads_files/molmo_d_7b.json')
    parser.add_argument('--num-selected-heads', type=int, default=3)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    molmo = MolmoModule(args)
    processor = MolmoProcessor.from_pretrained(args.vlm_checkpoint, trust_remote_code=True, device_map=args.device)
    # processor.image_processor = MolmoImageProcessor(
    #     max_crops=12,
    #     overlap_margins=(4, 4),
    #     base_image_input_size=(336, 336),
    #     image_token_length_w=12,
    #     image_token_length_h=12,
    #     image_patch_size=14,
    #     image_padding_mask=True,
    #     do_normalize=True,
    #     image_mean=(0.48145466, 0.4578275, 0.40821073),
    #     image_std=(0.26862954, 0.26130258, 0.27577711),
    # )
    inputs = processor.process(
        images=[Image.open('/home/sean/PartPointing/visualization/example.png')],
        text='Point to the cap of the bottle.'
    )
    inputs = {k: v.to(args.device).unsqueeze(0) for k, v in inputs.items()}
    
    # assistant_token_id = 21388
    with torch.no_grad():
        attentions = molmo.model.get_attentions(**inputs)
        attentions = torch.cat(attentions, 0)


    image_start_pos = (inputs['input_ids'][0] == special_token_dict['<im_start>']).nonzero(as_tuple=True)[0] + 1
    image_end_pos = (inputs['input_ids'][0] == special_token_dict['<im_end>']).nonzero(as_tuple=True)[0]
    assistant_id_pos = (inputs['input_ids'][0] == special_token_dict['ĠAssistant']).nonzero(as_tuple=True)[0]
    
    image_patch_pos = []
    num_cols = []
    for img_start, img_end in zip(image_start_pos, image_end_pos):
        image_patch_pos.append(inputs['input_ids'][0, img_start:img_end] == special_token_dict['<im_patch>'])
        num_cols.append(sum(inputs['input_ids'][0, img_start:img_end] == special_token_dict['<im_col>']))
    
    nail_thumb_attention = attentions[:, :, assistant_id_pos - 1, image_start_pos[0]:image_end_pos[0]][:, :, 0, image_patch_pos[0]]
    high_res_attention = attentions[:, :, assistant_id_pos - 1, image_start_pos[1]:image_end_pos[1]][:, :, 0, image_patch_pos[1]]
    
    l, h, _ = nail_thumb_attention.shape
    nail_thumb_attention = nail_thumb_attention.reshape((l, h, num_cols[0], -1))
    high_res_attention = high_res_attention.reshape((l, h, num_cols[1], -1))

    _, _, h, w = high_res_attention.shape
    nail_thumb_attention = F.interpolate(nail_thumb_attention, size=(h, w), mode='bilinear', align_corners=True)
    text_to_image_attentions = torch.zeros((h, w), device=nail_thumb_attention.device)
    selected_heads_list = [[19, 5], [16, 0], [11, 24]]
    
    for (l, h) in selected_heads_list:
        nail = gaussian_blur(nail_thumb_attention[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)
        high = gaussian_blur(high_res_attention[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)
        text_to_image_attentions += nail
        text_to_image_attentions += high
        # text_to_image_attentions += nail
    
    
    text_to_image_attentions = text_to_image_attentions.unsqueeze(0).unsqueeze(0)
    text_to_image_attentions = F.interpolate(text_to_image_attentions, size=(74, 74), mode='bilinear', align_corners=True)# .squeeze(0).squeeze(0)
    text_to_image_attentions = F.interpolate(text_to_image_attentions, size=(336, 336)).squeeze(0).squeeze(0).cpu().numpy()
    
    
    image_path = "/home/sean/PartPointing/visualization/example.png"
    image = Image.open(image_path)
    resized_image = image.resize((336, 336), Image.BICUBIC)
    img_np = np.array(resized_image)
    
    plt.imshow(img_np)
    plt.imshow(text_to_image_attentions, cmap='viridis', alpha=0.65)
    plt.axis('off')
    plt.tight_layout(pad=0)        
    plt.savefig("image_2_image.png")

import ast
import cv2
import json
import torch

from torchvision.transforms.functional import gaussian_blur
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import matplotlib.pyplot as plt

special_token_dict = {
    '<|vision_start|>': 151652,
    '<|vision_end|>':   151653,
    '<|image_pad|>':    151655,
    '<|im_end|>':       151645,
}

class Qwen2_5_VLModule():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.selected_heads_list = None
        self.load_model()

    def get_image_input_size(self, img_path):
        msg = [{"role":"user","content":[{"type":"image","image":img_path}]}]
        imgs, vids = process_vision_info(msg)
        txt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[txt], images=imgs, videos=vids, padding=True, return_tensors="pt")
        return inp['image_grid_thw'][0][1]*14, inp['image_grid_thw'][0][2]*14
    
    
    def load_model(self):
        if self.processor == None:
            self.processor = Qwen2_5_VLProcessor.from_pretrained(self.config.vlm_checkpoint, use_fast=True, device_map=self.config.device, min_pixels=256*28*28, max_pixels=1369*28*28)
        if self.model == None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.vlm_checkpoint, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=self.config.device)
        if self.selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                selected_heads_list = json.load(f)[:self.config.num_selected_heads]
            self.selected_heads_list = [(selected_heads['layer'], selected_heads['head']) for selected_heads in selected_heads_list]
    
    
    def prepare_inputs(self, image_file, query):
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"},],},
            {"role": "user", "content":
                [
                    {"type": "image", "image": image_file}, 
                    {"type": "text", "text": query},
                ]
            }
        ]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(message)
        inputs = self.processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")    
        return inputs
    
    
    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        input_height, input_width = self.get_image_input_size(image_file)
        
        # Prepare Input
        inputs = self.prepare_inputs(image_file, query)
        query_inputs = self.processor(text=[query], images=None, videos=None, padding=True, return_tensors="pt",)
            
        # Obtain Image Start Pos
        image_start_pos = torch.where(inputs.input_ids==special_token_dict['<|vision_start|>'])[1] + 1
        image_end_pos = torch.where(inputs.input_ids==special_token_dict['<|vision_end|>'])[1]

        # Obtain Query Start Pos
        query_start_pos = -1
        for i in range(len(inputs['input_ids'][0]) - len(query_inputs['input_ids'][0]) + 1):
            if torch.equal(inputs['input_ids'][0][i:i+len(query_inputs['input_ids'][0])], query_inputs['input_ids'][0]):
                query_start_pos = i
                break    
        assert query_start_pos != -1
            
        # Prefiliing
        with torch.no_grad():
            results = self.model(**inputs.to(self.config.device), output_attentions=True)

        attentions = torch.cat([results['attentions'][i].to(self.config.device) for i in range(len(results['attentions']))] ,0) # [Layer, Head, Query, Key]
        attentions = attentions[:, :, query_start_pos+len(query_inputs['input_ids'][0])-1, image_start_pos:image_end_pos]
        attentions = attentions.reshape(*attentions.shape[:2], input_height // 28, input_width // 28)
        text_to_image_attentions = torch.zeros((input_height // 28, input_width // 28), device=self.config.device)
        
        if self.selected_heads_list != None:
            for idx, (l, h) in enumerate(self.selected_heads_list):
                selected_score = gaussian_blur(attentions[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0) 
                text_to_image_attentions += selected_score
        else:
            text_to_image_attentions = attentions.sum(dim=(0, 1))
        
        
        text_to_image_attentions = (text_to_image_attentions - text_to_image_attentions.min()) / (text_to_image_attentions.max() - text_to_image_attentions.min())
        return text_to_image_attentions
    
    def get_patch_size(self):
        return 28
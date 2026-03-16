import ast
import cv2
import json
import torch

from torchvision.transforms.functional import gaussian_blur
from transformers import AutoModelForCausalLM

# try: 
#     from model.vlm_modules.ovis.modeling_ovis2_5 import Ovis2_5
# except:
#     from modeling_ovis2_5 import Ovis2_5

from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
special_token_dict = {
    '<|vision_start|>': 151652,
    '<|vision_end|>':   151653,
    '<|image_pad|>':    151655,
    '<|vision_pad|>':   151654,
    '<|im_end|>':       151645,
    '</':           522,
}

class Ovis2_5_VLModule():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.selected_heads_list = None
        self.question_template = '{query}'
        self.load_model()
    
    def load_model(self):
        if self.model == None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_checkpoint, 
                torch_dtype=torch.bfloat16,
                device_map=self.config.device, 
                trust_remote_code=True,
                attn_implementation='eager'
            )

        if self.selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                selected_heads_list = json.load(f)[:self.config.num_selected_heads]
            self.selected_heads_list = [(selected_heads['layer'], selected_heads['head']) for selected_heads in selected_heads_list]

    def get_image_input_size(self, img_path):
        msg = [{"role":"user","content":[{"type":"image","image":img_path}]}]
        imgs, vids = process_vision_info(msg)
        txt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[txt], images=imgs, videos=vids, padding=True, return_tensors="pt")
        return inp['image_grid_thw'][0][1]*14, inp['image_grid_thw'][0][2]*14
    
    def prepare_inputs(self, image_file, query):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_file)},
                {"type": "text", "text": self.question_template.format(query=query)},
            ],
        }]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=False,
            enable_thinking=False,
            max_pixels=1024 * 1024,
        )
     
        return input_ids, pixel_values, grid_thws
    
    
    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        while query[-1] == " ":
            query = query[:-1]
            
        input_ids, pixel_values, grid_thws = self.prepare_inputs(image_file, query)
        
        query_input_ids = torch.tensor(self.model._tokenize_with_visual_placeholder(query))
    
        query_start_pos = -1
        for i in range(len(input_ids[0]) - len(query_input_ids) + 1):
            if torch.equal(input_ids[0][i:i+len(query_input_ids)], query_input_ids):
                query_start_pos = i
                break    
        assert query_start_pos != -1
        
        img_pos = input_ids[0] == -300

        with torch.no_grad():
            results = self.model(
                input_ids=input_ids.to(self.config.device),
                pixel_values=pixel_values.to(self.config.device),
                grid_thws=grid_thws.to(self.config.device),
                attention_mask=None,
                output_attentions=True
            )

        attentions = torch.cat([results['attentions'][i].to(self.config.device) for i in range(len(results['attentions']))] ,0) # [Layer, Head, Query, Key]
        attentions = attentions[:, :, query_start_pos + len(query_input_ids) - 1, img_pos]
        attentions = attentions.reshape(*attentions.shape[:2], grid_thws[0][1] // 2, grid_thws[0][2] // 2)
        text_to_image_attentions = torch.zeros((grid_thws[0][1] // 2, grid_thws[0][2] // 2), device=self.config.device)
        
        if self.selected_heads_list != None:
            for (l, h) in self.selected_heads_list:
                selected_score = gaussian_blur(attentions[l, h].unsqueeze(0), kernel_size=[7, 7], sigma=1.0).squeeze(0)                
                text_to_image_attentions += selected_score
        else:
            text_to_image_attentions = attentions.sum(dim=(0, 1))
        
        text_to_image_attentions = (text_to_image_attentions - text_to_image_attentions.min()) / (text_to_image_attentions.max() - text_to_image_attentions.min())
        return text_to_image_attentions
    
    def get_patch_size(self):
        return 28
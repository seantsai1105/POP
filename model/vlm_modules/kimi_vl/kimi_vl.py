from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ast
import cv2
import json
from torchvision.transforms.functional import gaussian_blur
    
class Kimi_VLModule():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.selected_heads_list = None
        self.question_template = '{query}'
        self.load_model()
    
    
    def load_model(self):
        if self.processor == None:
            self.processor = AutoProcessor.from_pretrained(self.config.vlm_checkpoint, trust_remote_code=True)
        if self.model == None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_checkpoint,
                device_map=self.config.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation='eager',
            )
            
        if self.selected_heads_list == None:
            with open(f"{self.config.selected_heads_file}") as f:
                selected_heads_list = json.load(f)[:self.config.num_selected_heads]
            self.selected_heads_list = [(selected_heads['layer'], selected_heads['head']) for selected_heads in selected_heads_list]
    
    
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
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
        
        return inputs
    
    
    @torch.no_grad()
    def get_text_to_image_score(self, image_file, query):
        # Prepare Input
        inputs = self.prepare_inputs(image_file, query)
        
        img_pos = torch.where(inputs['input_ids'][0]==163605)[0]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, use_cache=False)
            
        attentions = torch.cat([outputs['attentions'][i].to(self.config.device) for i in range(len(outputs['attentions']))] , 0)
        l, h = attentions.shape[:2]
        attentions = attentions[:, :, -2, img_pos].reshape(l, h, inputs['image_grid_hws'][0][0] // 2, inputs['image_grid_hws'][0][1] // 2)

        text_to_image_attentions = torch.zeros((inputs['image_grid_hws'][0][0] // 2, inputs['image_grid_hws'][0][1] // 2), device=self.config.device)
        
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

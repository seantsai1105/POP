import os
import json
import random
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as maskUtils
from scipy.ndimage import label
from tqdm import tqdm
import torch.nn.functional as F
import pickle

from model.vision_backbone.dinov2_vit import Dinov2Module
from model.vision_backbone.dinov3_vit import Dinov3Module

from model.vlm_modules.qwen2_5_vl.qwen2_5_vl import Qwen2_5_VLModule
from model.vlm_modules.molmo.molmo import MolmoModule
from model.vlm_modules.internvl3.internvl3 import InternVL3_VLModule
from model.vlm_modules.ovis.ovis import Ovis2_5_VLModule
from model.vlm_modules.kimi_vl.kimi_vl import Kimi_VLModule

class FewShotPointer():
    def __init__(self, config):
        self.config = config
        
        # Loading VLM
        if 'qwen' in config.vlm_checkpoint.lower():
            print("Loading QwenVL2.5 Module")
            self.vlm_module = Qwen2_5_VLModule(config)
        if 'molmo' in config.vlm_checkpoint.lower():
            print("Loading Molmo Module")
            self.vlm_module = MolmoModule(config)
        if 'internvl3' in config.vlm_checkpoint.lower():
            print("Loading InternVL3 Module")
            self.vlm_module = InternVL3_VLModule(config)
        if 'ovis' in config.vlm_checkpoint.lower():
            print("Loading Ovis2.5 Module")
            self.vlm_module = Ovis2_5_VLModule(config)
        if 'kimi' in config.vlm_checkpoint.lower():
            print("Loading KimiVL Module")
            self.vlm_module = Kimi_VLModule(config)
        
        # Loading ViT
        if 'dinov2' in config.vision_backbone_checkpoint.lower():
            print("Loading DINOv2 Module")
            self.vision_backbone = Dinov2Module(config)
        if 'dinov3' in config.vision_backbone_checkpoint.lower():
            print("Loading DINOv3 Module")
            self.vision_backbone = Dinov3Module(config)
        
    def center_point(
        self,
        score_map,
        patch_size,
        ori_size,
        input_size
    ):
        ori_height, ori_width = ori_size
        input_height, input_width = input_size
        flat_idx = torch.argmax(score_map)
        max_index = torch.unravel_index(flat_idx, score_map.shape)
        y = int((max_index[0] * patch_size + patch_size // 2) * ori_height / input_height)
        x = int((max_index[1] * patch_size + patch_size // 2) * ori_width / input_width)
        return y, x
    
    def k_shot_pointing(self, 
        k, 
        target_image_file, 
        target_query,
        reference_image_file_list, 
        reference_point_list,
    ):
        assert k == len(reference_image_file_list)
        assert k == len(reference_point_list)
        
        # Obtain text-to-image score
        text_to_image_score = self.vlm_module.get_text_to_image_score(target_image_file, target_query)
        
        # Obtain image-to-image score
        image_to_image_score = self.vision_backbone.get_image_to_image_score(
            target_image_file,
            reference_image_file_list, 
            reference_point_list, 
        )
        
        # Fuse text-to-image and image-to-image
        H = W = self.vision_backbone.image_size // self.vision_backbone.patch_size
            
        text_to_image_score = text_to_image_score.unsqueeze(0).unsqueeze(0)
        text_to_image_score = F.interpolate(text_to_image_score, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        fused_score = (image_to_image_score * text_to_image_score).unsqueeze(0).unsqueeze(0)
        final_score =  F.interpolate(fused_score, size=(2*H, 2*W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                
        # Take center point of patch as prediction 
        ori_img = Image.open(target_image_file).convert("RGB")
        ori_width, ori_height = ori_img.size
        pred_point = self.center_point(final_score, self.vision_backbone.patch_size // 2, (ori_height, ori_width), (self.vision_backbone.image_size, self.vision_backbone.image_size))
                
        return pred_point
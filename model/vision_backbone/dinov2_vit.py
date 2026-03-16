import torch
import numpy as np
import cv2
import torch.nn.functional as F

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from pycocotools import mask as maskUtils



class Dinov2Module():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.model = None
        self.image_size = 896
        self.patch_size = 14
        self.load_model()
    
    
    def load_model(self):
        if self.processor == None:
            self.processor = AutoImageProcessor.from_pretrained(self.config.vision_backbone_checkpoint, device_map=self.config.device)
            self.processor.do_center_crop = False
            self.processor.crop_size = 896
            self.processor.do_resize = True
            self.processor.size = {"height": 896, "width": 896}
        if self.model == None:
            self.model = AutoModel.from_pretrained(
                self.config.vision_backbone_checkpoint,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16, 
                device_map=self.config.device).eval()
    
    @torch.no_grad()
    def get_reference_features(
        self,
        reference_image_file,
        reference_point,
    ):
        if type(reference_image_file) == str:
            reference_img = Image.open(reference_image_file).convert("RGB")
        else:
            reference_img = reference_image_file
        reference_width, reference_height = reference_img.size
        
        reference_inputs = self.processor(images=reference_img, return_tensors="pt").to(device=self.config.device)
        reference_outputs = self.model(**reference_inputs)
        reference_last_hidden_state = reference_outputs.last_hidden_state[0, 1:, :] 
        
        
        # Obtain a single patch from reference point
        y, x = reference_point
        x_patch = int(x / reference_width * self.image_size) // self.patch_size
        y_patch = int(y / reference_height * self.image_size) // self.patch_size
        reference_single_patch_feature = reference_last_hidden_state[int(y_patch * (self.image_size / self.patch_size) + x_patch)]
        
        return reference_last_hidden_state, reference_single_patch_feature
    
    @torch.no_grad()
    def compute_backward_score(
        self,
        target_last_hidden_state,
        reference_last_hidden_state,
        single_patch_feature, 
    ):
        target_last_hidden_state_norm = F.normalize(target_last_hidden_state, dim=1)
        reference_last_hidden_state_norm = F.normalize(reference_last_hidden_state, dim=1)
        
        similarity = target_last_hidden_state_norm @ reference_last_hidden_state_norm.T 

        best_idxs = similarity.argmax(dim=1)

        reference_matched = reference_last_hidden_state_norm[best_idxs]
        if single_patch_feature.dim() == 1:
            single_patch_feature = single_patch_feature.unsqueeze(0)
        single_patch_feature_norm = F.normalize(single_patch_feature, dim=1)
       
        scores = reference_matched @ single_patch_feature_norm.T
        return scores
    
    
    @torch.no_grad()
    def get_image_to_image_score(
        self,
        target_image_file,
        reference_image_file_list,
        reference_point_list,
    ):
        if type(target_image_file) == str:
            target_img = Image.open(target_image_file).convert("RGB")
        else:
            target_img = target_image_file
        target_width, target_height = target_img.size
        
        # Encoding
        target_inputs = self.processor(images=target_img, return_tensors="pt").to(device=self.config.device)
        target_outputs = self.model(**target_inputs)
        target_last_hidden_state = target_outputs.last_hidden_state[0, 1:, :] 
        
        
        image_to_image_score = torch.ones((self.image_size // self.patch_size, self.image_size // self.patch_size), device=self.config.device)
        
        for reference_image_file, reference_point in zip(reference_image_file_list, reference_point_list):
            reference_last_hidden_state, reference_single_patch_feature = self.get_reference_features(reference_image_file, reference_point)
            
            # Forward score
            single_patch_similarity = (F.normalize(target_last_hidden_state, dim=1) @ F.normalize(reference_single_patch_feature, dim=0).T).reshape(self.image_size // self.patch_size, self.image_size // self.patch_size)
            
            # Backward score
            backward_score = self.compute_backward_score(
                target_last_hidden_state,
                reference_last_hidden_state,
                reference_single_patch_feature
            ).reshape(self.image_size // self.patch_size, self.image_size // self.patch_size)
        
            combined_score = single_patch_similarity * backward_score
            combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min())
            image_to_image_score *= combined_score
        
        return image_to_image_score
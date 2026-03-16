import os
import json
import random
import torch
import argparse
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from pycocotools import mask as maskUtils
from model.few_shot_pointer import FewShotPointer
from PIL import Image

def get_inner_most_point(mask_path):
    binary_mask = np.array(Image.open(mask_path))
    dist_transform = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=5)
    y, x = np.unravel_index(np.argmax(dist_transform), binary_mask.shape)
    return y, x

def get_prompt(vlm_checkpoint, query):
    """
    Construct a simplified prompt for each VLM checkpoint to make it easier
    to locate the last token of the query during inference.
    """
    
    assert vlm_checkpoint in [
        'allenai/Molmo-7B-D-0924',
        'AIDC-AI/Ovis2.5-9B', 
        'Qwen/Qwen2.5-VL-7B-Instruct', 
        'OpenGVLab/InternVL3-8B', 
        'moonshotai/Kimi-VL-A3B-Instruct'
    ]
    
    if vlm_checkpoint == 'allenai/Molmo-7B-D-0924':
        return f'Point to the {query}'
    
    elif vlm_checkpoint == 'AIDC-AI/Ovis2.5-9B':
        return f'Point to the <ref>{query}'
    
    elif vlm_checkpoint == 'Qwen/Qwen2.5-VL-7B-Instruct':
        return f'Point to the {query}'
    
    elif vlm_checkpoint == 'OpenGVLab/InternVL3-8B':
        return f'Locate the region this sentence describes: <ref>{query}</ref>'
    
    elif vlm_checkpoint == 'moonshotai/Kimi-VL-A3B-Instruct':
        return f'Locate the {query}'

def main(args):
    # Loading data
    with open(f"{args.dataset}") as f:
        datas = json.load(f)
    
    with open(f"{args.support_set}") as f:
        example_datas = json.load(f)
        
    fsp = FewShotPointer(args)
    
    results = []
    random.seed(args.random_seed)
    for idx, data in enumerate(tqdm(datas)):
        obj, part = data['part_list'][0]['object'].replace(" ", "_"), data['part_list'][0]['part'].replace(" ", "_")
        img_path = os.path.join(args.image_root, 'test/images', data['image_path'].replace(".jpg", f"-{obj}-{part}.jpg"))
        
        candidates = [
            d for d in example_datas if 
            d['part_list'][0]['object'] == data['part_list'][0]['object'] and 
            d['part_list'][0]['part'] == data['part_list'][0]['part'] and
            d['image_path'] != data['image_path']
        ]
        
        if len(candidates) < args.k_shots: 
            continue
        
        example_data = random.choices(candidates, k = args.k_shots)
        
        reference_image_file_list = [os.path.join(args.image_root, 'train/train1800/images', d['image_path'].replace(".jpg", f"-{obj}-{part}.jpg")) for d in example_data]
        reference_point_list = [get_inner_most_point(os.path.join(args.image_root, 'train/train1800/masks', d['image_path'].replace(".jpg", f"-{obj}-{part}.png"))) for d in example_data]
        
        if data['part_list'][0]['object'].endswith("s"):
            target_query = "{object}' {part}".format(object = data['part_list'][0]['object'], part=data['part_list'][0]['part'])
        else:
            target_query = "{object}'s {part}".format(object = data['part_list'][0]['object'], part=data['part_list'][0]['part'])
        
        target_query = get_prompt(args.vlm_checkpoint, target_query)
        pred_point = fsp.k_shot_pointing(
            k = args.k_shots,
            target_image_file = img_path,
            target_query = target_query,
            reference_image_file_list = reference_image_file_list,
            reference_point_list = reference_point_list,
        )
        
        ground_truth_mask_path = os.path.join(args.image_root, 'test/masks', data['image_path'].replace(".jpg", f"-{obj}-{part}.png"))    
        ground_truth_mask = np.array(Image.open(ground_truth_mask_path))
        results.append(int(ground_truth_mask[pred_point])) # pred_point is in (y, x) format
        
    print(f"Accuracy: {sum(results) / len(results)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vlm-checkpoint', 
        type=str, 
        default='allenai/Molmo-7B-D-0924',
        choices=[
            'allenai/Molmo-7B-D-0924',
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'AIDC-AI/Ovis2.5-9B',
            'OpenGVLab/InternVL3-8B',
            'moonshotai/Kimi-VL-A3B-Instruct'
        ]
    )
    parser.add_argument(
        '--vision-backbone-checkpoint', 
        type=str, 
        default='facebook/dinov3-vitl16-pretrain-lvd1689m',
        choices=[
            'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'facebook/dinov2-large'
        ]
    )
    parser.add_argument('--selected-heads-file', type=str, default='./model/vlm_modules/vlm_selected_heads_files/molmo_d_7b.json')
    parser.add_argument('--num-selected-heads', type=int, default=3)
    parser.add_argument('--k-shots', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='./data/InstructPart/data_test.json')
    parser.add_argument('--support-set', type=str, default='./data/InstructPart/data_train.json')
    parser.add_argument('--image-root', type=str, default='./data/InstructPart/')
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
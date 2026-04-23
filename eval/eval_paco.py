import os
import json
import random
import argparse
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from pycocotools import mask as maskUtils
from model.few_shot_pointer import FewShotPointer

def get_inner_most_point(segmentation):
    binary_mask = maskUtils.decode(segmentation)
    dist_transform = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=5)
    y, x = np.unravel_index(np.argmax(dist_transform), binary_mask.shape)
    return y, x

def get_prompt(vlm_checkpoint, query, part):
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
        return f'For {query}, point to its {part}'
    
    elif vlm_checkpoint == 'AIDC-AI/Ovis2.5-9B':
        return f'For {query}, point to <ref>its {part}'
    
    elif vlm_checkpoint == 'Qwen/Qwen2.5-VL-7B-Instruct':
        return f'For {query}, point to its {part}'
    
    elif vlm_checkpoint == 'OpenGVLab/InternVL3-8B':
        return f'For {query}, locate the region this sentence describes: <ref>its {part}</ref>'
    
    elif vlm_checkpoint == 'moonshotai/Kimi-VL-A3B-Instruct':
        return f'For {query}, locate its {part}'
    
def main(args):
    random.seed(args.random_seed)
    # Loading data
    with open(f"{args.dataset}") as f:
        datas = json.load(f)
    
    with open(f"{args.support_set}") as f:
        example_datas = json.load(f)
        
    fsp = FewShotPointer(args)
    
    results = []
    for idx, data in enumerate(tqdm(datas)):
        img_path = os.path.join(args.image_root, data['image_file'])
        
        candidates = [d for d in example_datas[data['object'] + ":" + data['part']] if d['image_file'] != data['image_file']]
        if len(candidates) < args.k_shots: 
            continue

        example_data = random.choices(candidates, k = args.k_shots)
        
        reference_image_file_list = [os.path.join(args.image_root, d['image_file']) for d in example_data]
        reference_point_list = [get_inner_most_point(d['segmentation']) for d in example_data]
        
        target_query = get_prompt(args.vlm_checkpoint, data['query_string'], data['part'])
        pred_point = fsp.k_shot_pointing(
            k = args.k_shots,
            target_image_file = img_path,
            target_query = target_query,
            reference_image_file_list = reference_image_file_list,
            reference_point_list = reference_point_list,
        )
        
        ground_truth_mask = maskUtils.decode(data['segmentation'])
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
    parser.add_argument('--selected-heads-file', type=str, default=None,
        help='Path to selected-heads JSON; auto-inferred from --vlm-checkpoint if omitted.')
    parser.add_argument('--num-selected-heads', type=int, default=3)
    parser.add_argument('--k-shots', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='./data/PACO/paco_lvis_v1_test_part_anns_with_query.json')
    parser.add_argument('--support-set', type=str, default='./data/PACO/paco_lvis_v1_train_preprocessed.json')
    parser.add_argument('--image-root', type=str, default='./data/PACO/coco')
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
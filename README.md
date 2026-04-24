# Pointing at Parts: Training-Free Few-Shot Grounding in Multimodal LLMs

## Table of Contents

- [TL;DR](#tldr)
- [Install](#install)
- [Data Preparation](#data-preparation)
- [Evaluation](#evaluation)
- [Main Results](#main-results)
- [Citation](#citation)


## TL;DR

**POP** is a training-free, plug-and-play approach for part-level pointing in Multimodal LLMs under a few-shot setup. It fuses (1) language-guided attention maps extracted from the frozen MLLM with (2) visual semantic correspondences between the target image and labelled support examples, produced by a self-supervised vision encoder (DINOv3). Without any additional training, POP improves 1-shot part-pointing accuracy by up to **8.9 points** on pointing-capable MLLMs (Qwen2.5-VL, Ovis2.5, Molmo) and up to **30.9 points** on MLLMs without pointing post-training (InternVL3, Kimi-VL), averaged across PACO, InstructPart, and PartImageNet++.


## Install

1. Clone this repository
   ```bash
   git clone https://github.com/seantsai1105/POP.git
   cd POP
   ```

2. Create a conda environment and install the package
   ```bash
   conda create -n pop python=3.10 -y
   conda activate pop
   pip install -e .
   ```

## Data Preparation

### 1. Download Preprocessed Annotations
Download the preprocessed annotation files from our release: [data](https://github.com/seantsai1105/POP/releases/tag/data)

### 2. Download Images

Please download the original images for each dataset:

#### PACO-LVIS (PACO)
PACO uses images from the COCO-2017 dataset. Please download the images from the [official COCO website](https://cocodataset.org/#download).

#### InstructPart
Please follow the instructions in the [official repository of InstructPart](https://github.com/zifuwan/InstructPart) to prepare the dataset.

#### PartImageNet++
PartImageNet++ uses images from ImageNet-1K. Please download the images from the [official ImageNet website](https://www.image-net.org/challenges/LSVRC/2012/).

### 3. Dataset Hierarchy

<details><summary> After downloading all required files (annotations and images), please organize the project directory as follows: </summary>


```
project_root/
├── data/
│   ├── InstructPart/
│   │   ├── data_train.json
│   │   ├── data_test.json
│   │   ├── train/
│   │   │   ├── train1800/
│   │   │   │   ├── images/
│   │   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   │
│   ├── PACO/
│   │   ├── paco_lvis_v1_train_preprocessed.json
│   │   ├── paco_lvis_v1_test_part_anns_with_query.json
│   │   └── coco/
│   │       ├── train2017/
│   │       ├── val2017/
│   │       └── test2017/
│   │
│   └── PartImageNet++/
│       ├── category_name.json
│       ├── partimagenet++_train.json
│       ├── partimagenet++_val.json
│       └── ImageNet2012/
│           ├── train/
│           └── val/
│
├── eval/
└── model/
```

</details>


## Evaluation

Each dataset has its own entry-point script. The commands below run the default 1-shot setting with DINOv3-ViT-L/16 and Qwen2.5-VL-7B. Swap `--vlm-checkpoint` to use a different VLM (see [Supported Models](#supported-models)) — the matching `selected_heads_file` is picked up automatically.

### PACO-LVIS
```bash
python eval/eval_paco.py \
    --vlm-checkpoint Qwen/Qwen2.5-VL-7B-Instruct \
    --vision-backbone-checkpoint facebook/dinov3-vitl16-pretrain-lvd1689m \
    --k-shots 1 \
    --random-seed 0
```

### InstructPart
```bash
python eval/eval_instructpart.py \
    --vlm-checkpoint Qwen/Qwen2.5-VL-7B-Instruct \
    --vision-backbone-checkpoint facebook/dinov3-vitl16-pretrain-lvd1689m \
    --k-shots 1 \
    --random-seed 0
```

### PartImageNet++
```bash
python eval/eval_partimagenet++.py \
    --vlm-checkpoint Qwen/Qwen2.5-VL-7B-Instruct \
    --vision-backbone-checkpoint facebook/dinov3-vitl16-pretrain-lvd1689m \
    --k-shots 1 \
    --random-seed 0
```

### Supported Models

<table>
<tr>
<td width="45%" valign="top">

| VLM | `--vlm-checkpoint` |
|---|---|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Ovis2.5-9B | `AIDC-AI/Ovis2.5-9B` |
| Molmo-7B-D | `allenai/Molmo-7B-D-0924` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` |
| Kimi-VL-A3B | `moonshotai/Kimi-VL-A3B-Instruct` |

</td>

<td width="10%"></td> <!-- 👈 這個就是空一行的關鍵 -->

<td width="45%" valign="top">

| Vision Backbone | `--vision-backbone-checkpoint` |
|---|---|
| DINOv3-ViT-L/16 (default) | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| DINOv2-ViT-L/14 | `facebook/dinov2-large` |

</td>
</tr>
</table>

`--selected-heads-file` is inferred from `--vlm-checkpoint` automatically. To override, pass a path under `model/vlm_modules/vlm_selected_heads_files/`.


## Main Results

Part-pointing accuracy (%) in the 1-shot setting with DINOv3-ViT-L/16. See the paper for 0-shot, 3-shot, and ablations.

| VLM | PACO | InstructPart | PartImageNet++ |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 51.6 | 87.9 | 80.1 |
| Ovis2.5-9B | 53.0 | 88.5 | 81.7 |
| Molmo-7B-D | 55.8 | 90.9 | 84.1 |
| InternVL3-8B | 45.7 | 86.1 | 78.3 |
| Kimi-VL-A3B | 49.7 | 90.3 | 79.4 |


## Citation

Our paper will appear at CVPR 2026. The BibTeX below is a placeholder and will be updated once the proceedings are available.

```bibtex
```

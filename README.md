#  Pointing at Parts: Training-Free Few-Shot Grounding in Multimodal LLMs [CVPR 2026]
## Table of Contents

- [TL;DR](#tldr)
- [Install](#install)
- [Data Preparation](#data-preparation)
- [Evaluation](#Evaluation)


## TL;DR
## Install
1. Clone this repository
```bash
git clone https://github.com/seantsai1105/POP.git
cd POP
```

2. Install Package
```Shell
conda create -n pop python=3.10 -y
conda activate pop
pip install -r requirements.txt
```

## Data Preparation
### 1. Download Preprocessed Annotations
Download the preprocessed annotation files from our release:
[data](https://github.com/seantsai1105/POP/releases/tag/data)

### 2. Download Images

Please download the original images for each dataset:

#### PACO-LVIS (PACO)
PACO uses images from the COCO-2017 dataset. Please download the images from the [official COCO website](https://cocodataset.org/#download).

#### InstructPart
Please follow the instructions in the [official repository of InstructPart](https://github.com/zifuwan/InstructPart) to prepare the dataset.

#### PartImageNet++
PartImageNet++ uses images from ImageNet-10K. Please download the images from the [official ImageNet website](https://www.image-net.org/challenges/LSVRC/2012/).

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

## Citation

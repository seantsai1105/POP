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
[link_here]

### 2. Download Raw Images
Please download the original images for each dataset:

#### PACO
[official link]

#### InstructPart
[official link]

#### PartImageNet++
[official link]
## Dataset Structure

After downloading all required files (annotations and images), please organize the project directory as follows:
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
## Evaluation

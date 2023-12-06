# Towards More Practical Group Activity Detection:<br> A New Benchmark and Model

### [Dongkeun Kim](https://dk-kim.github.io/), [Youngkil Song](https://www.linkedin.com/in/youngkil-song-8936792a3/), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/), [Suha Kwak](https://suhakwak.github.io/)

### [Project Page](http://dk-kim.github.io/CAFE) | [Paper](https://arxiv.org/abs/2312.02878)

## Overview
This work introduces the new benchmark Café dataset and the new model Group Transformer. 

## Requirements

- Ubuntu 20.04
- Python 3.8.5
- CUDA 11.0
- PyTorch 1.7.1

## Conda environment installation
    conda env create --file environment.yml

    conda activate gad

    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    
## Install additional package
    sh scripts/setup.sh
   

## Download datasets (Subset of Cafe dataset only)

    sh scripts/download_datasets.sh

## Download trained weights

    sh scripts/download_checkpoints.sh

## Run test scripts

- Cafe dataset (split by view)
  
        sh scripts/test_cafe_view.sh

- Cafe dataset (split by place)
  

        sh scripts/test_cafe_place.sh

## Run train scripts

- Cafe dataset (split by view)


        sh scripts/train_cafe_view.sh

- Cafe dataset (split by place)


        sh scripts/train_cafe_place.sh



# File structure

    ├── Dataset/
    │     └── cafe/
    │           └── gt_tracks_24.pkl
    ├── dataloader/
    ├── evaluation/
    │     └── gt_tracks_24.txt
    ├── label_map/
    ├── models/
    ├── scripts/
    └── util/
    train.py
    test.py
    environment.yml
    README.md

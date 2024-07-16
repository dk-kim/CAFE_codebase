# Towards More Practical Group Activity Detection:<br> A New Benchmark and Model

### [Dongkeun Kim](https://dk-kim.github.io/), [Youngkil Song](https://www.linkedin.com/in/youngkil-song-8936792a3/), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/), [Suha Kwak](https://suhakwak.github.io/)

### [Project Page](http://dk-kim.github.io/CAFE) | [Paper](https://arxiv.org/abs/2312.02878)

## Overview
This work introduces the new benchmark dataset, Café, and a new model for group activity detection (GAD). 

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
   
## Download datasets

Download Café dataset from:   <br/> 

https://cvlab.postech.ac.kr/research/CAFE/

## Download trained weights

    sh scripts/download_checkpoints.sh

or from:   <br/>  
https://drive.google.com/file/d/1W_2gkzARCzSdK8Db4G4pkzN3GrJTYo8R/view?usp=drive_link

## Run test scripts

- Café dataset (split by view)
  
        sh scripts/test_cafe_view.sh

- Café dataset (split by place)
  

        sh scripts/test_cafe_place.sh

## Run train scripts

- Café dataset (split by view)


        sh scripts/train_cafe_view.sh

- Café dataset (split by place)


        sh scripts/train_cafe_place.sh


## File structure

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

## Citation
If you find our work useful, please consider citing our paper:
```BibTeX
@article{kim2023towards,
  title={Towards More Practical Group Activity Detection: A New Benchmark and Model},
  author={Kim, Dongkeun and Song, Youngkil and Cho, Minsu and Kwak, Suha},
  journal={arXiv preprint arXiv:2312.02878},
  year={2023}
}
```

## Acknowledgement
This work was supported by the NRF grant and the IITP grant funded by Ministry of Science and ICT, Korea (RS-2019-II191906, IITP-2020-0-00842, NRF-2021R1A2C3012728, RS-2022-II220264). 

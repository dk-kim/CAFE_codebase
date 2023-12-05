# Requirements

- Ubuntu 20.04
- Python 3.8.5
- CUDA 11.0
- PyTorch 1.7.1

# Conda environment installation
    conda env create --file environment.yml

    conda activate gad

    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    
# Install additional package
    sh scripts/setup.sh
   

# Download datasets (Subset of Cafe dataset only)

    sh scripts/download_datasets.sh

# Download trained weights

    sh scripts/download_checkpoints.sh

# Run test scripts

- Cafe dataset (split by view)
  

    sh scripts/test_cafe_view.sh

- Cafe dataset (split by place)
  

    sh scripts/test_cafe_place.sh

# Run train scripts

- Cafe dataset (split by view)


    sh scripts/train_cafe_view.sh

- Cafe dataset (split by place)


    sh scripts/train_cafe_place.sh



# File structure

── Dataset/ <br/>
│   │── cafe/ <br/>
│   │   └── gt_tracks_24.pkl <br/>
│── dataloader/ <br/>
│── evaluation/ <br/>
│   └── gt_tracks_24.txt <br/>
│── label_map/ <br/>
│── models/ <br/>
│── scripts/ <br/>
│── util/ <br/>
train.py <br/>
test.py <br/>
environment.yml <br/> 
README.md <br/> 
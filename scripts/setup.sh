wget https://github.com/longcw/RoIAlign.pytorch/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
cd RoIAlign.pytorch-master/
python setup.py install
mv roi_align/ roi_align_torch/
cd ../
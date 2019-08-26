# 3C-Net

## Overview
This package is a PyTorch implementation of our paper [3C-Net: Category Count and Center Loss for Weakly-Supervised Action Localization](https://arxiv.org/abs/1908.08216), to be published at ICCV 2019.
# 3C-Net

## Data
We use the same I3D features, for the Thumos14 and ActivityNet 1.2 datasets, released by [Sujoy Paul](https://github.com/sujoyp/wtalc-pytorch). The annotations are already included in this repository. 

## Training 3C-Net
The model can be trained using the following commands. See options.py for additional parse arguments

```javascript 
# Running on Thumos14 
python main.py --dataset-name Thumos14
# Running on ActivityNet 1.2
python main.py --dataset-name ActivityNet1.2 --activity-net --num-class 100
```

## Citation
Please cite the following work if you use this package.
```javascript
@article{narayan20193cnet,
  title={3C-Net: Category Count and Center Loss for Weakly-Supervised Action Localization},
  author={Narayan, Sanath and Cholakkal, Hisham and Shahbaz Khan, Fahad and Shao, Ling},
  journal={arXiv preprint arXiv:1908.08216},
  year={2019}
}
```

## Dependencies
This codebase was built on the W-TALC package found [here](https://github.com/sujoyp/wtalc-pytorch) and has the following dependencies.
1. PyTorch 0.4.1, Tensorboard Logger 0.1.0
2. Python 3.6
3. numpy, scipy among others


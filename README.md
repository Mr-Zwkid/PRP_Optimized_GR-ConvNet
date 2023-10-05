# Multimodal Grasping Detection based Optimized GR-ConvNet
Aiming at the robotic grasping detection problem in unstructured environments, the GR-ConvNet deep learning model is optimized by introducing the attention mechanism, adding dropout layers and designing the GS-Inception-Residual block and the multimodal fusion module. The migration learning mechanism is introduced during model training to improve the model generalization performance and shorten the training time. As the ablation experiments and multimodal fusion experiments show, targeted at Cornell dataset of image splitting and object splitting, the optimized GR-ConvNet achieves higher detection accuracy while ensuring the detection speed.

This repository contains GR-ConvNet from the paper, based on which we make the optimazation:

#### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/antipodal-robotic-grasping-using-generative/robotic-grasping-on-cornell-grasp-dataset)](https://paperswithcode.com/sota/robotic-grasping-on-cornell-grasp-dataset?p=antipodal-robotic-grasping-using-generative)

```
@inproceedings{kumra2020antipodal,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}}
}
```

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/Mr-Zwkid/PRP_Optimized_GR-ConvNet.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd PRP_Optimized_GR-ConvNet
$ pip install -r requirements.txt
```

## Datasets

This repository supports the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`


## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```


## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```


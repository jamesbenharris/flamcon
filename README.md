# Flamcon: Flamingo+Falcon Language-Video Cross Attention Network

Flamcon is a flexible and powerful deep learning architecture that enables efficient cross-modal attention between language, image, and video data. This repository contains the code for training and using Flamcon, along with detailed descriptions of its components and how to get started.

![flamcon image](data/flamcon.png)

## Table of Contents

- [Overview](#Overview)
- [Requirements](#Requirements)
- [Usage](#Usage)
  - [Training](#Training)
- [Configuration](#configuration)
- [License](#license)
- [Team](#Team)
- [Future plans](#Futureplans)
- [Acknowledgments](#Acknowledgments)
- [Citations](#Citations)

## Overview

Flamcon (Flamingo+Falcon Language-Video Cross Attention Network) is a versatile architecture that facilitates cross-modal attention between text and image inputs. It combines vision and language processing using transformer-based models to perform tasks such as text generation, image/video captioning, and more.

Key features of Flamcon:
- **Cross-Attention**: Flamcon enables efficient and effective cross-attention between language and image inputs, allowing the model to learn rich and meaningful associations.
- **Modular Design**: The architecture is modular, allowing you to easily configure the number of layers, attention heads, and other hyperparameters.
- **Efficient Training**: Leveraging the benefits of DeepSpeed and Fully Sharded Data Parallelism (FSDP), Flamcon is designed for efficient and scalable distributed training.
- **Video Support**: Video is supported with a training example(train.py) and WebVid dataloader. 

## Requirements

- Docker image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
- Hugging Face Transformers
- DeepSpeed (for distributed training with FSDP)
- tqdm (for progress bars)
- At least one A6000 GPU 48GB ram.

## Usage

### Infrastructure

The following hardware was used for testing and training the model.

1.  2 A6000 GPU's 48GB memory for development
2.  8 A6000 GPU's 48GB memory for training
3.  1 A6000 GPU 48GB memory for testing

### Data

The [10 million WebVid dataset](https://github.com/m-bain/webvid) was used to extract samples for training.  Each video was preprocessed into a 40 frame 256x256 resolution clip(40,3,256,256).  A text filter was then applied to select a more focused sample.  Finally, 10k videos were randomly select and split into training and a validation dataset. 

### Training

To train the Flamcon model, follow these steps:

1. Clone this repository: `git clone https://github.com/vrne/flamcon`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare your training data and adjust the data paths in the code.
    a.  See src/dataloader.py for examples.
4. Run the training script:
   ```
   torchrun --nnodes=1 --nproc_per_node=2 train.py --batch 6 --max_frames 40
   ```


### Testing

To test the Flamcon model once trained, follow these steps:

1. Run the testing script:
   ```
   python test.py
   ```
   or load demo/generate.ipynb
## Configuration

You can adjust various parameters in the training and inference scripts to customize the behavior of the Flamcon model. Refer to the script comments and the [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/) for more details on configuring distributed training.

## License

This project is licensed under the [MIT License](LICENSE).

## Future plans
- [ ] Train model on 10 million videos
- [ ] Train model on a custom dataset

## Team

Flamcon is developed by:

[Ben Harris](https://jamesbenjaminharris.com)

## Acknowledgments
This code is based on Lucidrains' [flamingo implementation](https://github.com/lucidrains/flamingo-pytorch) and [OpenFlamingo](https://github.com/mlfoundations/open_flamingo).  I appreciate your assistance in expediting the process by improving my understanding of fsdp wrapping and checkpointing techniques!

## Citations

```
@article{awadalla2023openflamingo,
  title={OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models},
  author={Anas Awadalla and Irena Gao and Josh Gardner and Jack Hessel and Yusuf Hanafy and Wanrong Zhu and Kalyani Marathe and Yonatan Bitton and Samir Gadre and Shiori Sagawa and Jenia Jitsev and Simon Kornblith and Pang Wei Koh and Gabriel Ilharco and Mitchell Wortsman and Ludwig Schmidt},
  journal={arXiv preprint arXiv:2308.01390},
  year={2023}
}
```

# Deformable Capsule Network for Object Detection

This project implements a Deformable Capsule Network (DeformCaps) for object detection using PyTorch, as described in the paper ["Deformable Capsules for Object Detection"](https://arxiv.org/pdf/2104.05031).

## Overview

Deformable Capsules (DeformCaps) are designed to address the challenge of object detection in computer vision. This implementation includes:

- A novel capsule structure (SplitCaps)
- A dynamic routing algorithm (SE-Routing)
- Training and evaluation on the MS COCO dataset

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/naivoder/DeformableCapsuleNetwork.git
    cd DeformableCapsuleNetwork
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Download and extract the MS COCO dataset:
    - This will be handled automatically by the script.

2. Run the training and evaluation script:

    ```sh
    python main.py
    ```

This will download the MS COCO dataset, train the DeforCaps Network, and evaluate its performance.

## Note

Make sure you have sufficient disk space and memory for downloading and processing the MS COCO dataset (~20 Gb).

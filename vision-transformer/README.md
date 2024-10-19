# Vision Transformers

This directory contains an implementation of Vision Transformers (ViTs), a powerful architecture for computer vision tasks. Vision Transformers leverage the Transformer model, originally designed for natural language processing, to process image data.

## Table of Contents

- [Vision Transformers](#vision-transformers)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Model Architecture
The Vision Transformer (ViT) model consists of the following components:

- Patch Embedding: Splits an image into fixed-size patches and projects them into a lower-dimensional embedding space.
- Transformer Encoder: Applies multiple layers of the Transformer encoder to process the sequence of patch embeddings.
- Classification Head: A fully connected layer applied to the [CLS] token for image classification tasks.

# Datasets
This repository supports various datasets for training and evaluation. The datasets are properly formatted and the paths are specified in the configuration files. Example datasets include:

- ImageNet
- CIFAR-10
- CIFAR-100

# Contributing
We welcome contributions to improve this project! Please follow these steps to contribute:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes.
- Push the changes to your fork.
- Create a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# References
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

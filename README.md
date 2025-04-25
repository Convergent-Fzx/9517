# Aerial Image Classification on SkyView Dataset

This repository contains code for aerial landscape image classification using the SkyView dataset. The project explores both traditional machine learning approaches and deep learning models, with additional experiments on handling imbalanced (long-tail) datasets.

## Dataset

The project uses the [SkyView dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset), which contains aerial images of various landscape categories. 

## Methods Implemented

### Feature Extraction
- Shift (image transformations)
- Local Binary Patterns (LBP)

### Traditional Machine Learning
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

### Deep Learning
- ResNet
- EfficientNet

### Imbalanced Data Handling
- Long-tail distribution experiments

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- pandas
- torchvision
- OpenCV
- scikit-learn
- scikit-image
- matplotlib
- seaborn
- jupyterlab

```bash
pip install torch torchvision opencv-python scikit-learn scipy matplotlib seaborn numpy scikit-image jupyterlab
```

## How to run 

1. Go into the project directory and make a directory called `res` to save the results
   
    ```bash
      makedir res
    ```

2. Change the `img_dir` in each file to the actual directory saving Aerial Landscapes dataset, if you have clone the repo from github then this is not necessary

3. Run those code blocks in jupyter notebooks

## Code Attribution

This project is primarily composed of original work by the team. However, several open-source libraries and references were used to support implementation and experimentation. Below is a list of tools and external resources used:
### Libraries & Tools

- [PyTorch](https://pytorch.org/) – Deep learning framework used to implement and train ResNet and DenseNet models.
- [Torchvision](https://pytorch.org/vision/stable/models.html) – Used for pretrained models and image transformations.
- [OpenCV](https://opencv.org/) – For implementing SIFT and other image processing utilities.
- [Scikit-learn](https://scikit-learn.org/stable/) – For traditional ML models such as SVM, KNN, and Random Forest.
- [Imbalanced-learn](https://imbalanced-learn.org/stable/) – Techniques used to handle class imbalance (e.g., oversampling).
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) – For data visualization.

### Reference Materials & Tutorials

- [SIFT Feature Extraction – OpenCV Docs](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)
- [LBP Feature Extraction – PyImageSearch](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)
- [Understanding ResNet – Towards Data Science](https://towardsdatascience.com/residual-networks-resnet-cf6843a5eefa)
- [DenseNet Paper (Huang et al., 2017)](https://arxiv.org/abs/1608.06993)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) – Referenced for understanding scaling techniques and architecture improvements over standard CNNs.
- [Handling Imbalanced Datasets – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/handling-imbalanced-dataset-in-machine-learning/)
- [Long-Tail Learning: A Decoupled Learning Framework for Long-Tailed Recognition (Zhou et al., 2020)](https://arxiv.org/abs/1910.09217) – Provided guidance on training models on imbalanced datasets effectively.
.

All external resources are used in accordance with their respective licenses and are credited accordingly in code comments where applicable.

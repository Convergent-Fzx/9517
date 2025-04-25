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
- ResNet architecture
- DenseNet architecture

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
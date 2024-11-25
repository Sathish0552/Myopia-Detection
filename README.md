# Myopia Detection Using Supervised and Unsupervised Learning

This project aims to detect **myopia** (nearsightedness) from eye images using two different machine learning techniques: **Supervised Learning** and **Unsupervised Learning**. Both approaches focus on classifying images into two categories: "Normal" and "Myopia". 

The **Supervised Learning** model is trained with labeled data using a **VGG19** architecture, while the **Unsupervised Learning** method utilizes **KMeans clustering** to classify images based on features without labeled data.

## Table of Contents
- [Introduction](#introduction)
- [Supervised Learning Approach](#supervised-learning-approach)
- [Unsupervised Learning Approach](#unsupervised-learning-approach)
- [Results](#results)
- [Prerequisites](#prerequisites)


## Introduction

Myopia, commonly known as nearsightedness, is a refractive error that causes distant objects to appear blurry. Detecting myopia early can help prevent further deterioration and improve eye health. This project explores two approaches for detecting myopia from images of the eye using machine learning:

- **Supervised Learning**: In this method, we train a **VGG19** model on a labeled dataset of myopic and normal eye images. The goal is to predict whether a given image represents a normal or myopic eye based on the features learned during training.
- **Unsupervised Learning**: This method involves extracting features from images and applying the **KMeans clustering** algorithm to group images into clusters. We then assign one of the clusters to myopia and the other to normal eyes.

Both approaches leverage **Convolutional Neural Networks (CNNs)** for feature extraction, but differ in the way they handle the data and learn from it.

## Supervised Learning Approach

In the supervised learning method, we use a **VGG19 model**, which is a deep convolutional neural network (CNN) originally designed for image classification tasks. The VGG19 model was chosen for its ability to extract rich features from images, making it ideal for identifying subtle patterns in medical images like those of the eye.

### Key Features of the Supervised Learning Approach:
- **VGG19 Model**: A pre-trained deep CNN model that was fine-tuned for the task of myopia detection.
- **Training**: The model is trained on labeled data, with each image being tagged as either "normal" (0) or "myopia" (1).
- **Loss Function**: Binary cross-entropy loss is used to measure the difference between the predicted and true labels.
- **Optimizer**: The Adam optimizer is used to minimize the loss during training.
- **Performance**: The model achieved an **accuracy of 99.45%** on the test set with a loss of **0.0877**.

#### Model Architecture
The VGG19 architecture consists of 19 layers, including 16 convolutional layers and 3 fully connected layers. It is known for its simplicity and deep architecture, which makes it effective for image classification tasks.

## Unsupervised Learning Approach

In the unsupervised learning method, we apply **KMeans clustering** to the extracted features of eye images. Unlike supervised learning, no labels are required for training. Instead, the model groups images based on their visual similarities. The KMeans algorithm assigns images to one of two clusters, and we assume one cluster corresponds to myopic eyes while the other represents normal eyes.

### Key Features of the Unsupervised Learning Approach:
- **InceptionV3 Model**: Used for feature extraction. This model is known for its efficiency and is able to capture detailed features of images.
- **KMeans Clustering**: A clustering algorithm that groups images into clusters based on the similarity of their features. The number of clusters is predefined (in this case, 2 clusters).
- **Performance**: The KMeans clustering method achieves **98% accuracy** on unseen data, almost matching the performance of the supervised model without the need for labeled data.

### KMeans Clustering Process
1. **Feature Extraction**: We first extract features from the images using the **InceptionV3** model. These features represent high-level image characteristics that are useful for clustering.
2. **Clustering**: We apply the **KMeans algorithm** to the extracted features. KMeans then groups the images into two clusters: one for normal eyes and one for myopic eyes.
3. **Prediction**: For new images, we predict the cluster to which they belong, and based on the cluster, classify them as either "normal" or "myopic".

## Results

### Supervised Learning Results:
- **Accuracy**: 99.45%
- **Loss**: 0.0877  
  The supervised model demonstrates exceptional performance, with near-perfect accuracy on the test set.

### Unsupervised Learning Results:
- **Accuracy**: 98%  
  The unsupervised model achieves nearly the same accuracy as the supervised approach, with the added advantage of not requiring labeled data.

Both approaches show strong potential for myopia detection, with the supervised model offering slightly higher accuracy due to its use of labeled data for training.

## Prerequisites

To run this project, you will need:

- **Python 3.x**

### Necessary libraries:
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `matplotlib`

You can install these dependencies using the following command:
```bash
pip install -r requirements.txt


# Painting Style Classification and Feature Visualization

### Description
ArtNet is a deep learning project that. 


### Network Architecture
DenseNet-121 is used as a backbone of the network followed by a classifier layers consisted of:
```
Global Average Pooling -> Dropout -> Batch Normalization -> Dense Layer
```
Dropout rate is set to 0.5 and softmax function is used as the final layer's activation function.

### Network Training
The model (DenseNet-121 plus the classifier) is trained via transfer learning approach. I used DenseNet-121 pre-trained with ImageNet dataset (from Tensorflow's implementation) as the starting network's weights and the classifier weights are initialized randomly.

When training, we freeze all the layers in the DenseNet-121 except some of the last convolutional layers ['conv5_block15_2_conv', 'conv5_block16_1_conv', 'conv5_block16_2_conv']

### Requirements
This project is written fully in Python. Thus, you need a Python environment in your computer and the following dependencies to install:
```
tensorflow==2.2.0 
streamlit==0.63.0
numpy==1.18.1
Pillow==7.2.0
```

### How to run the ArtNet app
Download or clone this repo and execute the following command in your command prompt:
```
streamlit run app.py
```



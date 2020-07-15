# Painting Style Classification and Feature Visualization

### Description
ArtNet is a deep learning project that. 

### Dataset Preparation and Preprocessing
Dataset used in this project is from Painter by Numbers which available on Kaggle. The dataset is a huge collection of paintings labelled by the painters name, style (movement), genre, and many more. I sorted and categorized the paintings by the style and aggregated it into 5 major modern movements or styles:
```

```
* Please note that I've no formal education in art history and curation, the aggregation of the style is purely based on my common art knowledge. Thus, if you spot something odd/wrong and have suggestions/critics regarding the dataset, please do tell me :) 

After we have the selected paintings for the final dataset. We need to make the network learn two major things 

The whole selected paintings is then normalized to [0,1] and transformed into square image by padding (we don't 


### Network Architecture
DenseNet-121 is used as a backbone of the network followed by a classifier layers consisted of:
```
Global Average Pooling -> Dropout -> Batch Normalization -> Dense Layer
```
Dropout rate is set to 0.5 and softmax function is used as the final layer's activation function.

### Network Training
The network (DenseNet-121 plus the classifier) is trained via transfer learning approach. I used DenseNet-121 pre-trained with ImageNet dataset (from Tensorflow's implementation) as the starting network's weights and the classifier weights are initialized randomly.

When training, we freeze all the layers in the DenseNet-121 except some of the last convolutional layers to allow the network to learn some probable unique high-level features in paintings and the batch normalization layers.
```
['conv5_block15_2_conv', 'conv5_block16_1_conv', 'conv5_block16_2_conv']
```
The network is trained with cross entropy loss function (in keras/tensorflow: ```categorical_crossentropy```), Adam optimizer, and batch size of 32. The learning rate starting value is 0.0001 and decaying half after 2 epochs.

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



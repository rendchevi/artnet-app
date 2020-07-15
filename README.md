# Painting Style Classification and Feature Visualization

### Description
ArtNet is a deep learning project that. 


### Network Architecture
DenseNet-121 is used as a backbone of the network followed by a classifier layers consisted of:
```
Global Average Pooling -> Dropout -> Batch Normalization -> Dense Layer
```

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



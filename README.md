# ArtNet - Painting Style Classification and Feature Visualization

### Description
ArtNet is a deep learning 

I intend to share the training and preprocessing codes soon.

### How to install 
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

### Dataset Preparation, Preprocessing, and Augmentation
**Dataset Preparation**
Dataset used in this project is from Painter by Numbers which available on Kaggle. The dataset is a huge collection of paintings (about 23k unique paintings) labelled by the painters name, style (movement), genre, and many more. I sorted and categorized the paintings by the style and aggregated it into 5 major modern movements or styles:
```
# Painting Style Classes Aggregation
Cubism        -> ['Cubism', 'Tubism', 'Cubo-Expressionism', 'Mechanistic Cubism', 'Analytical Cubism', 'Cubo-Futurism', 'Synthetic Cubism']
Impressionism -> ['Impressionism', 'Post-Impressionism', 'Synthetism', 'Divisionism', 'Cloisonnism']
Expressionism -> ['Expressionism', 'Neo-Expressionism', 'Figurative Expressionism', 'Fauvism']
Realism       -> ['Realism', 'Hyper-Realism', 'Photorealism', 'Analytical Realism', 'Naturalism']
Abstract      -> ['Abstract Art', 'New Casualism', 'Post-Minimalism', 'Orphism', 'Constructivism', 'Lettrism', 'Neo-Concretism', 'Suprematism',
                 'Spatialism', 'Conceptual Art', 'Tachisme', 'Post-Painterly Abstraction', 'Neoplasticism', 'Precisionism', 'Hard Edge Painting']
```   

*Please note that I've no formal education in art history and curation, the aggregation of the style is purely based on my common art knowledge.*
*Thus, if you spot something odd/wrong and have suggestions/critics regarding the dataset, please do tell me :)*

**Dataset Preprocessing and Augmentation**
After we have the selected paintings for the final dataset. We need to make the network learn two things from the dataset, how the whole image information represents certain style and how texture details such as brush stroke and color tone represents certain style. 

To achieve the first one, we simply pad (we should maintain image's aspect ratio) and resize the the image into square with size ```[256x256x3]``` (the network input size). For the latter, we divide the image into 5 regions by cropping them without resize, to capture brush stroke texture and color in high-resolution. This process will augment the dataset by 5 folds. The final images in the dataset is around 58k images and divided into training, validation, and testing data in 75:15:15 ratio.

![Patch Image](https://github.com/rendchevi/artnet-app/blob/master/assets/patch_sample.jpg)

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

### Accuracy of the trained model
The trained model uploaded in this repo has a categorical accuracy around 66% after 15 epochs of training.
| Training Acc.  | Validation Acc. | Test Acc. |
| -------------- | --------------- | --------- |
| 65.54          | 66.16           | 66.75     |

![Validation Plot](https://github.com/rendchevi/artnet-app/blob/master/assets/plot_acc.png)

### Additional Information
- Preprocessing process was done in my local machine
- Training process was done in Google Colab with GPU (I intend to share the training code soon!)

### Future Improvements
- [ ] Host the app on the web
- [ ] Re-train on a bigger dataset to achieve higher accuracy
- [ ] Further research on multilabel classification for classifying art style/movement

# ArtNet - Painting Style Classification and Feature Visualization

### Description
ArtNet is a deep learning project with the goal to classify or identify modern painting styles and visualize what the network "sees" when making that decision.


The neural network (currently) learns 5 major art movements: **Cubism, Impressionism, Expressionism, Realism, and Abstract**. I chose these movements because their disctinction mainly presents in visual appereances such as the brush strokes, painting techniques, textures, and color tones. Unlike movements such as Dada and Surrealism, which distinction lies in the uncommon object and narration presented in the paintings. Thus, can make the network harder to learn and easier to misclassify with other styles (I've tried it and the results not really good, maybe on another project).  

I also thought it would be interesting to visualize what the network learned from classifying painting styles/movements, although it hasn't been optimized yet, I added option in the app to visualize what the network sees at some layers of the network. **GradCAM** is used as the feature visualization method, you can examine the code inside ```utils.py``` in ```get_heatmap()``` function, feel free to tinker with it for better visualization output, I haven't pay much attention to it.  

This repository is only containing the app to run the model on your own and the pre-trained model, I plan to upload the training and data preprocessing codes soon in different repository.

### How to install 
This project is written fully in Python. Thus, you need a Python environment in your computer and the following dependencies to install:
```
tensorflow==2.2.0 (or tensorflow-cpu is fine)
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

### Network Architecture and Training
DenseNet-121 is used as a backbone of the network followed by a classifier layers consisted of:
```
Global Average Pooling -> Dropout -> Batch Normalization -> Dense Layer
```
Dropout rate is set to 0.5 and softmax function is used as the final layer's activation function.

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
- [ ] Revamp the feature visualization options
- [ ] Re-train on a bigger dataset to achieve higher accuracy
- [ ] Further research on multilabel classification for classifying art style/movement

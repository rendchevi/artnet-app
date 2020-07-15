# ======================================================================
# Utilities Function for ArtNet - Art Style Classification and Visualization
# Author: Rendi Chevi
#         https://github.com/rendchevi
# ======================================================================

import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model

DISPLAY_WIDTH = 325

# ---> HELPER FUNCTIONS

def square_pad(img, img_size):
    # -------------------------------------
    # Function to pad image to square image
    # Input: Image, np.array
    # Output: Padded square image, tf.Tensor
    # -------------------------------------
    w, h, c = img.shape
    if w > h:
        img = tf.image.resize_with_pad(img, w, w)
        img = tf.image.resize(img, img_size)
    else:
        img = tf.image.resize_with_pad(img, h, h)
        img = tf.image.resize(img, img_size)

    return img

def get_heatmap(grad, img_target, img_size):
    # -------------------------------------
    # Function to apply GradCAM methods
    # Input: set of gradients per conv layer target, list of tf.Tensor; image target, np.array; image size, list
    # Output: composite image, np.array; heatmap image, np.array
    # -------------------------------------

    # Compute weight, ak, iterating trhough feature maps or channel of gradient
    w_ak = np.array([tf.reduce_mean(grad[:,:,j], axis = None) for j in range(grad.shape[-1])])
    # Get heatmap
    heatmap = np.array([w * grad[:,:,j] for w, j in zip(w_ak, range(grad.shape[-1]))])
    heatmap = np.mean(heatmap, axis = 0)
    # Apply ReLU
    heatmap[heatmap < 0] = 0
    # Normalize heatmap
    heatmap = heatmap / np.max(heatmap)
    # Convert to RGB image (still grayscale colored)
    heatmap = tf.repeat(tf.expand_dims(heatmap, axis = -1), 3, axis = -1)
    # Resize or upsample to match image target's size
    heatmap = tf.image.resize(heatmap, img_size)
    # Get composite image
    comp = np.multiply(img_target, heatmap)

    return comp, heatmap

# ---> CACHED FUNCTIONS

@st.cache(allow_output_mutation = True, show_spinner = False)
def initiate_cached_var():
    # -------------------------------------------
    # Function to initialize "cached" variables
    # -------------------------------------------
    np.save('cached/button_state.npy', [False, False, False, False])
    np.save('cached/gradients.npy', None)
    np.save('cached/pred_score.npy', np.array([[0],[0],[0],[0],[0]]))
    np.save('cached/hmaps_composite_1.npy', np.ones((3,DISPLAY_WIDTH,DISPLAY_WIDTH,3)))
    np.save('cached/hmaps_composite_2.npy', np.ones((3,DISPLAY_WIDTH,DISPLAY_WIDTH,3)))
    np.save('cached/hmaps_composite_3.npy', np.ones((3,DISPLAY_WIDTH,DISPLAY_WIDTH,3)))

@st.cache(allow_output_mutation = True, show_spinner = False)
def load_model(do_gradcam = False):
    # -------------------------------------
    # Function to load ArtNet model
    # Input: boolean option to extract intermediate layers for GradCAM
    # Output: ArtNet model in keras.Model
    # -------------------------------------

    # Build the ArtNet model
    artnet = tf.keras.models.load_model('model/artnet', compile = False)
    # Option to extract intermediate layers for GradCAM
    if do_gradcam == True:
        target_layer = ['conv2_block2_2_conv' , 'conv2_block6_2_conv'  , 'conv3_block8_2_conv',
                        'conv3_block12_2_conv', 'conv4_block16_2_conv' , 'conv4_block20_2_conv',
                        'conv4_block24_2_conv', 'conv5_block10_2_conv' , 'conv5_block16_2_conv',]
        model = Model(inputs = artnet.input, outputs = [artnet.output] + [artnet.get_layer(layer).output for layer in target_layer])
    else:
        model = Model(inputs = artnet.input, outputs = artnet.output)

    return model

@st.cache(allow_output_mutation = True, show_spinner = False)
def feed_forward_model(model, input_img):
    # -------------------------------------
    # Function to load ArtNet model
    # Input: boolean option to extract intermediate layers for GradCAM
    # Output: ArtNet model in keras.Model
    # -------------------------------------

    # Extract gradients of output classification with respect to targetted conv layer
    with tf.GradientTape() as tape:
        # Feed forward model
        outputs = model(tf.expand_dims(input_img, axis = 0))
        # Get outputs
        pred_score = outputs[0]
        conv_output = outputs[1:]
        # Get best class prediction
        loss = pred_score[:, np.argmax(pred_score)]
    pred_score = pred_score.numpy()
    gradients = tape.gradient(loss, conv_output)
    # Save gradients and prediction score as "cached" variables
    np.save('cached/pred_score.npy', pred_score)
    np.save('cached/gradients.npy', gradients)

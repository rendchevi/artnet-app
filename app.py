# ======================================================================
# GUI for ArtNet - Art Style Classification and Feature Visualization
# Author: Rendi Chevi
#         https://github.com/rendchevi
# ======================================================================

import streamlit as st
import numpy as np
from PIL import Image

import utils

DISPLAY_WIDTH = 325

# ------------------->
# APP INITIALIZATION
# ------------------->

# Display a wait message until the model is loaded
with st.spinner('Wait a sec, loading the ArtNet model!'):
    utils.initiate_cached_var()
    label_names = ['Cubism', 'Expressionism', 'Impressionism', 'Realism', 'Abstract']
    artnet = utils.load_model(do_gradcam = True)

# ------------->
# APP INTERFACE
# ------------->

# ===============
# Header content
# ===============
st.title('What Neural Network Sees in Art')
st.write('Visualize what a neural network sees when looking and classifying a painting')
button_state = np.load('cached/button_state.npy', allow_pickle = True)

# ===============
# Sidebar content ---> 1. Image Uploading Features
# ===============
uploaded_img = st.sidebar.file_uploader('Upload a Painting...', type = ['png', 'jpg', 'jpeg'])
if uploaded_img != None:
    # Load and normalize the image
    uploaded_img = np.array(Image.open(uploaded_img))[:,:,:3] / 255.0
    img_size_ori = uploaded_img.shape
    # Pad image to square
    uploaded_img = utils.square_pad(uploaded_img, img_size = [DISPLAY_WIDTH, DISPLAY_WIDTH])
    # Display the image
    display_img = uploaded_img.numpy()
    display_img[display_img == 0] = 1
    st.image(display_img, width = DISPLAY_WIDTH, caption = 'Input Image')

# ===============
# Sidebar content ---> 2. Run Prediction on Input Image
# ===============
button_run = st.sidebar.button('Predict Painting Style')
# Define input image
input_img = uploaded_img
if button_run and uploaded_img != None:
    button_state[0] = True
    # Feed image to compute gradients and predictions
    utils.feed_forward_model(artnet, input_img)

# Load computed predictions and gradients
pred_score = np.load('cached/pred_score.npy', allow_pickle = True)
gradients = np.load('cached/gradients.npy', allow_pickle = True)

# Display predictions
st.sidebar.text('{}    \t\t\t   {}%'.format(label_names[0], int(pred_score.flatten()[0] * 100)))
st.sidebar.progress(float(pred_score.flatten()[0]))
st.sidebar.text('{} \t\t\t   {}%'.format(label_names[1], int(pred_score.flatten()[1] * 100)))
st.sidebar.progress(float(pred_score.flatten()[1]))
st.sidebar.text('{} \t\t\t   {}%'.format(label_names[2], int(pred_score.flatten()[2] * 100)))
st.sidebar.progress(float(pred_score.flatten()[2]))
st.sidebar.text('{} \t\t\t   {}%'.format(label_names[3], int(pred_score.flatten()[3] * 100)))
st.sidebar.progress(float(pred_score.flatten()[3]))
st.sidebar.text('{} \t\t\t   {}%'.format(label_names[4], int(pred_score.flatten()[4] * 100)))
st.sidebar.progress(float(pred_score.flatten()[4]))

# ===============
# Sidebar content ---> 3. Visualize What the Model Sees
# ===============
hmaps_composite_1 = np.load('cached/hmaps_composite_1.npy', allow_pickle = True)
hmaps_composite_2 = np.load('cached/hmaps_composite_2.npy', allow_pickle = True)
hmaps_composite_3 = np.load('cached/hmaps_composite_3.npy', allow_pickle = True)

# Process and display 1st option of feature visualization via gradCAM
if button_state[0] == True:
    button_sees_1 = st.sidebar.button('See what the neural network sees')
    if button_sees_1:
        button_state[1] = True
        for i, grad in enumerate(gradients[6:]):
            grad = grad[0]
            comp, hmap = utils.get_heatmap(grad, input_img, img_size = [DISPLAY_WIDTH, DISPLAY_WIDTH])
            hmaps_composite_1[i] = comp
        st.image([hmaps_composite_1[0], hmaps_composite_1[1], hmaps_composite_1[2]],
                  width = int(DISPLAY_WIDTH / 1.5), caption = ['1','2','3'])
        np.save('cached/hmaps_composite_1.npy', hmaps_composite_1)

# Process and display 2nd option of feature visualization via gradCAM
if button_state[1] == True:
    button_sees_2 = st.sidebar.button('Go Deeper')
    if button_sees_2:
        button_state[2] = True
        for i, grad in enumerate(gradients[3:6]):
            grad = grad[0]
            comp, hmap = utils.get_heatmap(grad, input_img, img_size = [DISPLAY_WIDTH, DISPLAY_WIDTH])
            hmaps_composite_2[i] = comp
        st.image([hmaps_composite_1[0], hmaps_composite_1[1], hmaps_composite_1[2],
                  hmaps_composite_2[0], hmaps_composite_2[1], hmaps_composite_2[2]],
                  width = int(DISPLAY_WIDTH / 1.5), caption = ['1','2','3'] + ['4','5','6'])
        np.save('cached/hmaps_composite_2.npy', hmaps_composite_2)

# Process and display 3rd option of feature visualization via gradCAM
if button_state[2] == True:
    button_sees_3 = st.sidebar.button('Much Deeper')
    if button_sees_3:
        button_state[3] = True
        for i, grad in enumerate(gradients[:3]):
            grad = grad[0]
            comp, hmap = utils.get_heatmap(grad, input_img, img_size = [DISPLAY_WIDTH, DISPLAY_WIDTH])
            hmaps_composite_3[i] = comp
        st.image([hmaps_composite_1[0], hmaps_composite_1[1], hmaps_composite_1[2],
                  hmaps_composite_2[0], hmaps_composite_2[1], hmaps_composite_2[2],
                  hmaps_composite_3[0], hmaps_composite_3[1], hmaps_composite_3[2]],
                  width = int(DISPLAY_WIDTH / 1.5), caption = ['1','2','3'] + ['4','5','6'] + ['7','8','9'])
        np.save('cached/hmaps_composite_3.npy', hmaps_composite_3)

np.save('cached/button_state.npy', button_state)

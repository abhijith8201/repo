"""
UI app for Visual Question Answering(VQA) case study using streamlit.
"""


import os
import re
import joblib
import shutil
import contractions
import gdown

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import vgg16, resnet50


# creating folder to store featurizers and models
if not os.path.exists('featurizers'):
    os.mkdir('featurizers')
if not os.path.exists('model_save'):
    os.mkdir('model_save')

# download featurizers and models that is stored in drive
if len(os.listdir('featurizers')) <= 3:
    url = "https://drive.google.com/drive/folders/1AtBoIeSPz9Wr9e4Pv3Zbp2-DAX6wf4hj"
    gdown.download_folder(url, quiet=False, use_cookies=False)

if len(os.listdir('model_save')) <= 1:
    url = "https://drive.google.com/drive/folders/10Jlg0jpomdboLZNh2yejlDEpwHfxDRra"
    gdown.download_folder(url, quiet=False, use_cookies=False)


# import tokenization - The tokenization.py file can be found here: https://github.com/google-research/bert/blob/master/tokenization.py
from featurizers import tokenization


MAX_SEQUENCE_LENGTH = 22
IMG_TARGET_SIZE = (224, 224)


# Function to clean the questions
def clean_text(text):
    """
    Clean the text by fixing contractions, lowering the text and removing 
    any special characters.
    """
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text
    

def encode_questions_model_1(tknizr, embd_layer, text, seq_len):
    """Encode text data.

    Parameters:
    -----------
    tknizr: tensorflow.keras.preprocessing.text.Tokenizer
        Tokenizer to convert text to int sequences.

    embd_layer: tensorflow.keras.layers.Embedding
        tf keras Embedding layer to embed sequence.

    text: str
        Question to encode.
    
    seq_len: int
        max sequence length to pad the sequence to.

    Returns:
    --------
    encoded_sequence: tf.Tensor
        Encoded text sequence
    """
    # clean the text
    text = clean_text(text)
    # converting to int sequences using the tokenizer
    encoded_seq = tknizr.texts_to_sequences([text]) # need to pass list of values
    # padding sequences to seq_len
    encoded_seq = pad_sequences(
        encoded_seq, maxlen=seq_len, dtype='int32', padding='post'
    )
    # embed using the Embedding layer
    encoded_seq = embd_layer(encoded_seq)
    return encoded_seq


def preprocess_images_model_1(img_path, target_size):
    """Preprocess images.

    Parameters:
    -----------
    img_path: str 
        Path to image file

    target_size: tuple(height, width)
        Size to load the image.

    Returns:
    --------
    img_arr: np.ndarray
        Preprocessed image as np array
    """
    img = image.load_img(img_path, target_size=target_size)
    img_arr = image.img_to_array(img)
    # The images are converted from RGB to BGR, 
    # then each color channel is zero-centered with respect to the ImageNet dataset, 
    # without scaling.
    img_arr = vgg16.preprocess_input(img_arr)
    return img_arr


def encode_images_model_1(vgg_featurizer, img_path, target_size):
    """Encode image data.

    Parameters:
    -----------
    vgg_featurizer: tensorflow.keras.models.Model
       Model from which the activations of last hidden layer in vgg16 can be extracted.

    img_path: str
        Path to image file.
    
    target_size: tuple(height, width)
        Size to load the image.

    Returns:
    --------
    vgg_feats: tf.Tensor
        Activations from last hidden layer of vgg16 model.
    """
    # preprocess the image
    img_arr = preprocess_images_model_1(img_path, target_size)
    # vgg_featurizer
    vgg_feats = vgg_featurizer(np.expand_dims(img_arr, axis=0))
    return vgg_feats


def produce_ans_model_1(img_file, question):
    """
    Produce the answer for a question asked about an image using model_1.
    
    Parameters:
    -----------
    img_file: str or bytes
        Image file on which question is asked.
    
    question: str
        Question to which the model should produce answer.
    
    Returns:
    --------
    ans: str
        Answer produced by the model to the question asked about the image.
    """
    # encode the questions using the "encode_questions_model_1" funtion defined earlier
    enc_seq = encode_questions_model_1(
        data['tknizr'], data['glove_embedding_layer'], question, MAX_SEQUENCE_LENGTH)[0]
    # encode the images using the "encode_images_model_1" funtion defined earlier
    vgg_feats = encode_images_model_1(
        data['vgg_featurizer'], img_file, IMG_TARGET_SIZE)[0]
    # generate answer
    y_pred_val = data['model_1']([np.asarray([enc_seq]), np.asarray([vgg_feats])])
    y_pred_val = data['ohe'].inverse_transform(tf.one_hot(tf.argmax(y_pred_val, axis=1), depth=len(data['ohe'].categories_[0])))[0][0]
    return y_pred_val


st.title("Visual Question Answering (VQA)")  # title


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model_and_featurizers():
    """
    Load trained model and data feturizer objects for prediction. 
    """
    data = {}

    data['ohe'] = joblib.load("featurizers/ohe.joblib")
    data['model_1'] = load_model('model_save/best_model_1.h5')

    # Tokenizer
    data["tknizr"] = joblib.load("featurizers/tknizr.joblib")
    # indices for each of the word in the vocab
    WORD_INDEX = data["tknizr"].word_index
    # vocab size (no. of all tokens in the text(questions) train data)
    VOCAB_SIZE = len(WORD_INDEX) + 1 # +1 because all the indices we get are starting from 1 but after padding we also add 0 so total=len(word_index)+1

    # ==============================================================================
    # Text Embedding layer
    # load the whole GLOVE embedding into memory
    # Ref: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    EMBEDDING_DIM = 300

    embeddings_index = dict()
    f = open('featurizers/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Embedding layer
    data['glove_embedding_layer'] = Embedding(
        VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], 
        input_length=MAX_SEQUENCE_LENGTH, name='GLOVE_embedding_layer', 
        trainable=False
    )

    # ==============================================================================
    # VGG16 Featurizer. 
    # The weights obtained are a result of training on 'imagenet' dataset. 
    vgg16_model = vgg16.VGG16(
        weights='imagenet', 
        input_shape=IMG_TARGET_SIZE+ (3,), #`input_shape` must be a tuple of three integers. The input must have 3 channels if weights = 'imagenet'
    )
    vgg16_model.trainable = False

    # Create a model that outputs activations from vgg16's last hidden layer
    data['vgg_featurizer'] = Model(
        inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)
    data['vgg_featurizer'].trainable = False
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading necessary stuff...')
data = load_model_and_featurizers()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading necessary stuff...done!')


col1, col2 = st.columns(2, gap="medium")

with col1:
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width='always')

with col2:
    st.subheader("Question: ")
    question = st.text_area('Question: ').strip()

    submit_btn = st.button('Submit')
    if submit_btn:
        ans = produce_ans_model_1(uploaded_file, question)
    else:
        ans = ""

    st.subheader("Answer: ")
    st.markdown("**"+ans+"**")

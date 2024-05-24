# streamlit run TextSentimentApp.py

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# for crunching github data
import requests
import json
import time
import datetime
import os
import fnmatch
import pandas as pd

# NLP Packages
from textblob import TextBlob
import random
import time

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

from keras.layers import LSTM


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
# a load function with
# @st.cache
# list out the loading weight functionalities for the models :
@st.cache
def load_intermediate():
    files = find("final_model",  os.getcwd())
    # model=keras.models.load_model("/content/adam_acc_8643.h5")
    model = keras.models.load_model(files[0])
    return model


def model_predection(text_input):
    blob = TextBlob(str(text_input))
    return (blob.sentiment.polarity+1)/2


def get_fixed_word_to_id_dict():
    INDEX_FROM = 3   # word index offset

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v+INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id[" "] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    return word_to_id


def decode_to_sentence(data_point):
    NUM_WORDS = 10000  # only use top 1000 words

    word_to_id = get_fixed_word_to_id_dict()

    id_to_word = {value: key for key, value in word_to_id.items()}
    return ' '.join(id_to_word[id] for id in data_point)


def encode_sentence(sent):
    # print(sent)
    encoded = []

    word_to_id = get_fixed_word_to_id_dict()

    for w in sent.split(" "):
        if w in word_to_id:
            encoded.append(word_to_id[w])
        else:
            encoded.append(2)        
    return encoded



def predictor_page():
    # this page will be responsible for dealing with the predictions

    # decorations
    html_temp = """
    <div style="background-color:#000000;padding:10px">
    <h2 style="color:white;text-align:center;"><b> Mood Detection Predictor </b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""<br>""", unsafe_allow_html=True)

    # Multiple model need to delete
    #MODELS = {"Basic": model_predection, "Intermediate": model_KerasIntermediate,
    #          "Complex": model_KerasIntermediate}
    
    MODEL = model_predection
    # allow choice of models
    #option = st.selectbox(
    #    'Model Complexity:', list(MODELS.keys()))
    # print(MODELS.keys)
    current_model = MODEL
    #current_model = MODELS[option]
    

    random_text_string = "Hello World"
    # warming up the model

    results = current_model(random_text_string)
    # run the model load functions here:

    # create text input
    text_label = "Please Enter Your Text"
    filled_text = " "
    ip = st.text_input(text_label, filled_text) 
    show_preds = False
    # print(ip)

    show_preds = False if ((ip == filled_text) or (ip == "")) else True
    # print(show_preds)

    rm = """
    # for instant predictions or lazy predictions
    # perform predictions instantly or wait for user to press a button
    inst = st.checkbox("Instantaneous Predictions", True)
    if inst is False:
        preds=st.button("Predict")
    """

    # show results
    if (show_preds == True):
        positivity_scale = current_model(ip)
        # perform thresholding methods
        p = positivity_scale
        result = ["Hateful or Sad" if p <= 0.25 else "Demoralizing or Tense" if p <=
                  0.50 else "Neutral or Clam" if p <= 0.75 else "Overwhelming or Cheerful"]
        result = str(result[0]) + " Mood"
        st.success('Predicted Mood : {}'.format(result))

        filling_up_bar = """
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(int(positivity_scale)):
            latest_iteration.text(f'Measured Positivity : {i+1}')
            bar.progress(i + 1)
        """
        latest_iteration = st.empty()
        latest_iteration.text('Measured Percentage : ' +
                              str(positivity_scale*100) + " %")
        bar = st.progress(positivity_scale)

        st.balloons()



def writer(time=0, json_dict=None):
    with open('temp_data/data'+str(time)+'.txt', 'w') as outfile:
        json.dump(json_dict, outfile)




def sidebar_page(pages, header_img=None, title=None, text=None):
    if header_img != None:
        st.sidebar.markdown(header_img, unsafe_allow_html=True)


def sidebar_nav():

    # NAV BAR
    st.sidebar.markdown("""## Navigation Bar: <br> """, unsafe_allow_html=True)
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    current_page = st.sidebar.radio(
        " ", ["Predictions"])

    sidetext = """
    
    """

    st.sidebar.markdown(sidetext, unsafe_allow_html=True)

    all_pages = {"Predictions": predictor_page
                  }
    # predictor_page()

    func = all_pages[current_page]
    func()



ROOT_DIRECTORY = os.getcwd()
sidebar_nav()

# Add chatbot component at the bottom left corner
chatbot_html = """
<!-- Begin Tawk.to Script -->
<script type="text/javascript">
var Tawk_API=Tawk_API||{}, Tawk_LoadStart=new Date();
(function(){
var s1=document.createElement("script"),s0=document.getElementsByTagName("script")[0];
s1.async=true;
s1.src='https://embed.tawk.to/YOUR_TAWKTO_WIDGET_ID/default';
s1.charset='UTF-8';
s1.setAttribute('crossorigin','*');
s0.parentNode.insertBefore(s1,s0);
})();
</script>
<!-- End Tawk.to Script -->
"""
components.html(chatbot_html)
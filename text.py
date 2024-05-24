import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import requests
import json
import time
import datetime
import os
import fnmatch
import pandas as pd
from textblob import TextBlob
import random
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

@st.cache_data
def load_intermediate():
    files = find("final_model", os.getcwd())
    model = keras.models.load_model(files[0])
    return model

def model_predection(text_input):
    blob = TextBlob(str(text_input))
    return (blob.sentiment.polarity + 1) / 2

def get_fixed_word_to_id_dict():
    INDEX_FROM = 3
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id[" "] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    return word_to_id

def decode_to_sentence(data_point):
    NUM_WORDS = 10000
    word_to_id = get_fixed_word_to_id_dict()
    id_to_word = {value: key for key, value in word_to_id.items()}
    return ' '.join(id_to_word[id] for id in data_point)

def encode_sentence(sent):
    encoded = []
    word_to_id = get_fixed_word_to_id_dict()
    for w in sent.split(" "):
        if w in word_to_id:
            encoded.append(word_to_id[w])
        else:
            encoded.append(2)
    return encoded

def predictor_page():
    html_temp = """
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px;">
    <h2 style="color:white;text-align:center;"><b> Mood Detection Predictor </b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""<br>""", unsafe_allow_html=True)
    MODEL = model_predection
    current_model = MODEL
    random_text_string = "Hello World"
    results = current_model(random_text_string)
    text_label = "Please Enter Your Text"
    filled_text = " "
    ip = st.text_input(text_label, filled_text)
    show_preds = False
    show_preds = False if ((ip == filled_text) or (ip == "")) else True
    if show_preds:
        positivity_scale = current_model(ip)
        p = positivity_scale
        result = ["Hateful", "Demoralizing", "Neutral", "Calm", "Overwhelming", "Cheerful"]
        if p <= 0.2:
            mood = result[0]
        elif p <= 0.4:
            mood = result[1]
        elif p <= 0.5:
            mood = result[2]
        elif p <= 0.6:
            mood = result[3]
        elif p <= 0.8:
            mood = result[4]
        else:
            mood = result[5]
        st.success('Predicted Mood : {}'.format(mood))
        latest_iteration = st.empty()
        latest_iteration.text('Measured Percentage : ' + str(positivity_scale * 100) + " %")
        bar = st.progress(positivity_scale)
        st.balloons()

def about_page():
    html_temp = """
    <style>
    .reportview-container {
        background: url("https://example.com/about_background.jpg");
        background-size: cover;
    }
    </style>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.title("About")
    st.write("""
        This application is a Mood Detection Predictor built using NLP. After cleaning up the 
        texts using many cleaning procedures we use lemining and steaming to get better results.
        We use NLP techniques to analyze the sentiment of the text inputted by the user 
        and predicts the mood associated with the text.
        
        **Features:**
        - Input text and get mood predictions.
        - Uses neural network for sentiment analysis.
        - Uses models like LSTM, Bidirectional and GRU.
        - Visual feedback with progress bar and balloons.
        
        **Author:**
        - Developed by Our Team
        
        **Github:**
        -
    """)

def writer(time=0, json_dict=None):
    with open('temp_data/data' + str(time) + '.txt', 'w') as outfile:
        json.dump(json_dict, outfile)

def sidebar_page(pages, header_img=None, title=None, text=None):
    if header_img != None:
        st.sidebar.markdown(header_img, unsafe_allow_html=True)

def sidebar_nav():
    st.sidebar.markdown("""## Navigation Bar: <br> """, unsafe_allow_html=True)
    current_page = st.sidebar.radio("Navigation", ["Predictions", "About"])
    all_pages = {"Predictions": predictor_page, "About": about_page}
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
components.html(chatbot_html, height=0)

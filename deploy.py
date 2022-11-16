import numpy as np
import pandas as pd
import tensorflow as tf
import afinn
from afinn import Afinn
from joblib import dump, load
from keras.models import Sequential, load_model
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
import streamlit as st

def add_bg_from_url(link):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({link});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



tf.device('/cpu:0')

afinn=Afinn(language='en')

from nltk.corpus import stopwords

# Neural Network model for Review

model2 = load_model('model.h5')

# Neural Network Tokenizer
clf = load('filename.joblib')



# DataSet
def data_string(reviews,stars):
    data=pd.DataFrame({"Reviews":[reviews],"Stars":[stars]})
    return data


# NN Tokenizer
def update_tokenizer_nn(test_n):
    test_n=test_n[0].split(" ")
    for i in test_n:
        if i not in list(clf.index_word.values()):
            clf.fit_on_texts([i])



## Pipeline for Neural Network

def custom_pipeline_nn(data):
    import re
    import unicodedata
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    accented_chars=lambda x:remove_accented_chars(x)
    data['Reviews'].apply(accented_chars)
    import string
    def clean_text(text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub("[0-9" "]+"," ",text)
        text = re.sub('[‘’“”…]', '', text)
        return text
    data["Reviews"]=data["Reviews"].apply(lambda x: clean_text(x))
    def word_count(text):
        return len(text.split())
    stop = stopwords.words('english')
    data["Reviews"]=data["Reviews"].apply(lambda x: " ".join(x for x in x.split()  if x not in stop))
    update_tokenizer_nn(data["Reviews"])
    
    sent_length=70
    embedded_docs=pad_sequences(clf.texts_to_sequences(data["Reviews"]),padding='pre',maxlen=sent_length)
    
    return embedded_docs

#data=data_string("worst bad","5")

#st.write(f"Model output {model2.predict(custom_pipeline_nn(data))}")


with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("⭐⭐⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐","⭐⭐","⭐")
    )

### Interface part

with open("design.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

st.title("Welcome to Sentiment Analysis...")
st.subheader("Sentiments will be Analyzed based on Machine Learning Model.")

add_bg_from_url(link="https://m.media-amazon.com/images/S/aplus-media-library-service-media/207646d3-7456-4936-9e7d-1d106d099ab9.__CR0,0,1464,600_PT0_SX1464_V1___.jpg")


form=st.form("Enter details:", clear_on_submit=True)
review_title = form.text_input("Review Title:")
reviews = form.text_area("Enter Review here for Analysis:")
if form.form_submit_button("Analyze"):
    st.success("Submitted for Sentiment Analysis.....")
    st.text(f"Your Entered Review title:\n{review_title}\n")
    st.text(f"Your Entered Review title:\n{reviews}")
else:
    st.stop()


data=data_string(review_title.join(reviews),len(add_radio))



predicted_value=model2.predict(custom_pipeline_nn(data))
#st.write(f"Model output {predicted_value}")
frame=pd.DataFrame(predicted_value,columns=["Negative","Neutral","Positive"])
frame["Review_Sentiment"]=frame.idxmax(axis=1)
value=frame["Review_Sentiment"].values
st.subheader(f"Sentiment of review is {value[0]}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from joblib import load
import spacy
import time
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_lg")

__model = None

# Set page config
st.set_page_config(page_title="Sentiment Analysis Web App", layout="wide")

# for preprocessing
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# vectorise
def vectorize(text):
    return nlp(text).vector.reshape(1, -1)

# for finding the bias
def get_bias(text):
    clean_text = preprocess(text)
    embeddings = vectorize(clean_text)
    prediction = __model.predict(embeddings)[0]

    if prediction == -1:
        return "Negative"
    elif prediction == 1:
        return "Positive"
    else:
        return "Neutral"

# for loading the model
def load_saved_model():
    global __model
    try:
        with open("model.joblib", "rb") as f:
            __model = load(f)
        st.success("Model loaded successfully.")
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Main function to run the Streamlit app
def main():
    load_saved_model()

    st.markdown("""
        <style>
            .main {
                background-color: #2E2E2E;
                color: #E0E0E0;
                font-family: 'Arial', sans-serif;
            }
            .stButton>button {
                background-color: #FFA726;
                color: white;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                cursor: pointer;
                border: none;
                border-radius: 5px;
            }
            h1 {
                font-size: 36px;
                color: #FFA726;
            }
            h2 {
                font-size: 30px;
                color: #FFA726;
            }
            h3 {
                font-size: 24px;
                color: #FFA726;
            }
            h4 {
                font-size: 20px;
                color: #FFA726;
            }
            h5 {
                font-size: 18px;
                color: #FFA726;
            }
            h6 {
                font-size: 16px;
                color: #FFA726;
            }
            p {
                font-size: 14px;
                color: #E0E0E0;
            }
            .sidebar .sidebar-content {
                background-color: #424242;
                color: #E0E0E0;
            }
            .sidebar .sidebar-content h2 {
                color: #FFA726;
            }
        </style>
        """, unsafe_allow_html=True)

    st.title('Sentiment Analysis Web App')
    
    st.header('Home Page')
    st.subheader('Welcome to the Sentiment Analysis Web App!')
    st.markdown("""
        <div style="background-color:#FFA726;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">Analyze the sentiment of your text</h3>
        </div>
        <br>
    """, unsafe_allow_html=True)
    st.write('This app predicts sentiment (Positive, Negative, Neutral) based on user input.')
    st.write("""
        Sentiment analysis is a powerful tool used to determine the emotional tone behind a body of text. 
        It is commonly used in marketing to gauge public opinion, monitor brand and product reputation, 
        and understand customer experiences. This application allows you to enter any text and get an 
        instant sentiment prediction.
    """)
    st.write("""
        **How it works:**
        - **Step 1:** Enter the text you want to analyze in the input box below.
        - **Step 2:** Click on the 'Predict' button.
        - **Step 3:** See the predicted sentiment displayed on the screen.
    """)
    
    st.header('Predict Sentiment')
    text = st.text_area('Enter your comment here:', height=200, placeholder="Type your text here...")
    if st.button('Predict'):
        if __model is not None:
            prediction = get_bias(text)
            if prediction == "Positive":
                st.success(f'Predicted Sentiment: {prediction}')
            elif prediction == "Negative":
                st.error(f'Predicted Sentiment: {prediction}')
            else:
                st.info(f'Predicted Sentiment: {prediction}')
        else:
            st.error("Model not loaded. Please check the model file.")

    st.header('About')
    st.markdown("""
        <div style="background-color:#FFA726;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">About This App</h3>
        </div>
        <br>
    """, unsafe_allow_html=True)
    st.write('This is a sentiment analysis web application.')
    st.write('Created by R3X')

if __name__ == '__main__':
    main()

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Set up the title and description for the Streamlit app
st.title('Sentiment Analysis using LSTM')
st.write("This app uses a pre-trained LSTM model to classify the sentiment of the given text input.")

# Load the pre-trained LSTM model
@st.cache_resource  # Use cache to optimize loading time
def load_model():
    return tf.keras.models.load_model('sentiment_model.h5')

model = load_model()

# Load the tokenizer used during training
@st.cache_resource  # Cache the tokenizer as well
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tokenizer()

# Define the max sequence length (same as used during model training)
max_len = 100

# Function to predict sentiment
def predict_sentiment(text, tokenizer, max_len):
    # Preprocess the text using the tokenizer
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Get the prediction from the model
    prediction = model.predict(text_pad)[0][0]
    
    # Debugging: print the raw prediction
    st.write(f"Raw model prediction: {prediction}")

    # Return sentiment based on threshold
    if prediction >= 0.5:
        return 'Positive'
    else:
        return 'Negative'

# Text input box for the user to enter the text
user_input = st.text_area("Enter the text for sentiment analysis", "")

# Button to predict sentiment
if st.button('Analyze Sentiment'):
    if user_input:
        sentiment = predict_sentiment(user_input, tokenizer, max_len)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")

# Option to upload a text file for sentiment analysis
uploaded_file = st.file_uploader("Or upload a text file for sentiment analysis", type=["txt"])

if uploaded_file is not None:
    # Read the uploaded file
    text_data = uploaded_file.read().decode("utf-8")
    
    # Display the content of the file
    st.write("File content:")
    st.write(text_data)
    
    # Perform sentiment analysis
    sentiment = predict_sentiment(text_data, tokenizer, max_len)
    st.write(f"Sentiment: {sentiment}")

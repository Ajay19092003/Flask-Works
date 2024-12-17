from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


app = Flask(__name__)

# Parameters for tokenization and model
max_words = 5000
max_len = 100

# Sample data for illustration purposes
data = pd.DataFrame({
    'Review Text': [
        'This movie was fantastic! I loved every bit.',
        'Terrible movie. Waste of time.',
        'The plot was interesting, but the acting was weak.',
        'A masterpiece! One of the best movies ever made.',
        'I didn’t like the film. It was boring and too slow.'
    ],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
})

# Preprocessing: Tokenize text data
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['Review Text'])
X = tokenizer.texts_to_sequences(data['Review Text'])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# Build LSTM model
model = Sequential()

# Adding Embedding Layer
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))

# Adding LSTM Layer
model.add(LSTM(128, return_sequences=False))

# Adding Dropout for regularization
model.add(Dropout(0.2))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Fit the model
history = model.fit(X_pad, y, epochs=5, batch_size=32, validation_split=0.2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    seq = tokenizer.texts_to_sequences([review])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(pad_seq)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return jsonify({'review': review, 'sentiment': sentiment})

if __name__ == "__main__":
    app.run(debug=True)

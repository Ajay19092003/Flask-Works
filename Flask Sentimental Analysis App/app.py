from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join('model', 'sentiment_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')  # Render the main HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received review: {data['review']}")  # Debug output
        review = data['review']
        
        # Vectorize the input review
        review_vectorized = vectorizer.transform([review])
        
        # Make prediction
        prediction = model.predict(review_vectorized)
        print(f"Prediction: {prediction[0]}")  # Debug output
        
        return jsonify({'sentiment': prediction[0]})  # Return the sentiment prediction
    except Exception as e:
        print(f"Error: {e}")  # Log any error
        return jsonify({'error': str(e)}), 500  # Return error message

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app

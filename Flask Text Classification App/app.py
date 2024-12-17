from flask import Flask, request, jsonify, render_template_string
import joblib
import logging

app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model and vectorizer with error handling
try:
    model = joblib.load('model/classifier.pkl')  # Load the MultinomialNB model
    vectorizer = joblib.load('model/vectorizer.pkl')  # Load the TfidfVectorizer
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {e}")
    exit(1)

# HTML template for the local interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Predictor</title>
</head>
<body>
    <h1>Sentiment Prediction App</h1>
    <form action="" method="post">
        <label for="review">Enter a Review:</label><br>
        <textarea id="review" name="review" rows="4" cols="50"></textarea><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction is not none %}
        <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        review = request.form.get('review')
        if not isinstance(review, str) or not review.strip():
            logging.warning("Invalid or missing review text.")
            prediction = "Error: Review text must be a non-empty string."
        else:
            try:
                # Transform the review text using the vectorizer
                transformed_review = vectorizer.transform([review])

                # Make a prediction using the pre-loaded model
                result = model.predict(transformed_review)
                prediction = int(result[0])
                logging.info(f"Prediction result: {prediction}")
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                prediction = "Error: An issue occurred during prediction."
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

# app.py
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("model/knn_model.pkl")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form data
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

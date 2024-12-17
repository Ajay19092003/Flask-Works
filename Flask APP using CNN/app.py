from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the path to the model
model_path = 'model/your_cnn_model.h5'

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the pre-trained CNN model
model = load_model(model_path)

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the homepage
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = Image.open(filepath)
        img = img.resize((64, 64))  # Resize image to match model input size
        img = np.array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class to the actual label
        classes = ['class1', 'class2', 'class3']  # Replace with actual class names
        result = classes[predicted_class]

        return render_template('upload.html', prediction=result)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)

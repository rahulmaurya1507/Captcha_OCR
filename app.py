# Import necessary libraries
import os
import time
import base64
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image

from flask import Flask, render_template, request

from tensorflow.keras.models import model_from_json


# Valid file types expected to be uploaded.
FILE_TYPES = ['jpg', 'jpeg', 'png']
# Model weight and architecture directory
MODEL_DIR = 'models'
# Training image shape
TRAINING_IMG_SHAPE = (190, 80)

# Function for loading model architecture
def load_model():
    # Read the JSON file containing the model architecture
    with open(f'{MODEL_DIR}/ocr_model.json', 'r') as json_file:
        model_architecture = json_file.read()

    # Create a model from the loaded architecture and load pre-trained weights
    model = model_from_json(model_architecture)

    model.load_weights(f'{MODEL_DIR}/ocr_100_epoch_0.001_lr.h5')
    print('model loading complete')
    return model


# Function for loading and processing the image.
def process_image(files):
    images = []
    # Iterate through each file and process the image
    for file in files:
        # Open and convert the image to grayscale
        image = Image.open(file).convert('L')
        # Resizing the image to the shape of training images
        image = image.resize(TRAINING_IMG_SHAPE)
        # Normalize pixel values to the range [0, 1]
        image = np.array(image) / 255.0
        # Reshape the image for model input
        image = image.reshape(-1, image.shape[0], image.shape[1], 1)
        images.append(image)
    # Stack the processed images into a single array
    images = np.vstack(images)
    return images


model = load_model()


# Function for making predictions
def predictor(files):
    # Load the model
    # Process the images
    images = process_image(files)
    # Make predictions using the model and logging prediction time.
    start_time = time.time()
    prediction = model.predict(images)
    end_time = time.time()
    print('time taken: ', end_time - start_time)
    # Get the predicted labels
    labels = np.argmax(prediction, axis=-1)
    # Convert labels to a readable format
    labels = ['label: ' + ' '.join(map(str, label)) for label in labels]
    return labels


# Convert files to base64 encoding
def files2base64(files):
    return [base64.b64encode(file.read()).decode('utf-8') for file in files]

# Initialize the Flask application
app = Flask(__name__)
# Set maximum content length for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if files are present in the request
    if 'files' not in request.files:
        return render_template('index.html', message='No file part')

    # Get the list of files from the request
    files = request.files.getlist('files')

    # Filtering the files that are valid image files
    files = [file for file in files if file.filename.split('.')[-1] in FILE_TYPES]

    # If no valid file is uploaded, render the index.html template with an appropriate message
    if not files:
        return render_template('index.html', message='No valid file selected!!')

    # Convert files to base64 encoding
    base64_list = files2base64(files)
    # Make predictions and get labels
    labels = predictor(files)
    # Combine results for rendering
    results = [{'image_b64': image_b64, 'label': label} for image_b64, label in zip(base64_list, labels)]
    return render_template('result.html', results=results)

# Run the Flask application if this script is the main module
if __name__ == '__main__':
    app.run(debug=True)

import tensorflow as tf
import numpy as np
from google.cloud import storage
from PIL import Image
from flask import request
# from io import BytesIO
import tempfile

# Confg GCS
GCS_BUCKET = 'a_d_m'
GCS_MODEL_FILE = 'Appple_diesases_model.h5'
storage_client = storage.Client()


model = None

# Load model func
def load_model():
    global model
    if model is None:
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_FILE)
        model_bytes = blob.download_as_bytes()
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_filename = tmp.name

        # Ładowanie modelu do zmiennej
        model = tf.keras.models.load_model(tmp_filename)
    return model

# Google Cloud Functions
def predict_disease(request):
    global model

    # Strona HTML do przesyłania obrazów
    upload_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Apple Leaf Diseases Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body, html {
            margin: 0;
            padding: 0;
            height: 100%;
            position: relative;
            font-family: 'Roboto', sans-serif;
        }
        body::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('https://storage.cloud.google.com/background_adm/background.png');
            background-size: cover;
            opacity: 0.5;
            z-index: -1;
        }
        .content {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        input[type="file"], input[type="submit"] {
            font-size: 1.2em;
            padding: 10px 20px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
        }
    </style>
</head>
<body>
    <div class="content">
        <h1>Apple Leaf Diseases Detector</h1>
        <form action="" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Upload Image">
        </form>
    </div>
</body>
</html>

    '''

    # GET: return HTML
    if request.method == 'GET':
        return upload_html

    # POST: process of the uploaded image
    elif request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return 'No file provided', 400

        # loading model (if is not already loaded)
        model = load_model()

        # Process image
        image = Image.open(file.stream).resize((128, 128))
        image_array = np.array(image)
        img_batch = np.expand_dims(image_array, 0)

        # Clasification
        CLASS_NAMES = ['alternaria', 'apple_scab', 'black_rot', 'cedar_apple_rust', 'healthy']
        predictions = model.predict(img_batch)
        prediction = predictions[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        confidence_rounded = round(confidence)

        # Returning prediction
        return f"Predicted Class: {predicted_class}, Confidence: {confidence_rounded}%"

    return 'Method not allowed', 405

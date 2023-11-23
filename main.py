import tensorflow as tf
import numpy as np
from google.cloud import storage
from PIL import Image
from flask import request
from io import BytesIO

# Confg GCS
GCS_BUCKET = 'model_apple_diseases_2'
GCS_MODEL_FILE = 'model_apple_diseases_2/4'
storage_client = storage.Client()


model = None

# Load model func
def load_model():
    global model
    if model is None:
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_FILE)
        model_bytes = blob.download_as_bytes()
        model = tf.keras.models.load_model(BytesIO(model_bytes))
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
    </head>
    <body>
        <h1>Apple Leaf Diseases Detector</h1>
        <form action="" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Upload Image">
        </form>
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
        confidence = float(np.max(prediction))

        # Returning prediction
        return f"Predicted Class: {predicted_class}, Confidence: {confidence}"

    return 'Method not allowed', 405

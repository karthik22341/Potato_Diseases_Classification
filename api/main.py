from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn

import base64
from PIL import Image
app = FastAPI()

# Mounting static files for serving CSS, JS, and other static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model
MODEL = tf.keras.models.load_model('saved_models/1.keras')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Upload Image</title>
            <link href="/static/style.css" rel="stylesheet" type="text/css">
        </head>
        <body>
            <h1>Upload Image</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


import base64
from PIL import Image


@app.post('/predict', response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Perform prediction using the loaded model
    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Encode the image as base64
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

    # Render the HTML template with the prediction result and base64-encoded image
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <style>
            /* Your CSS styles here */
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Result</h1>
            <img src="data:image/png;base64,{}" alt="Uploaded Image">
            <div class="result">
                <p class="label">Predicted Class:</p>
                <p>{}</p>
            </div>
            <div class="result">
                <p class="label">Confidence:</p>
                <p>{}</p>
            </div>
            <a href="/">Upload Another Image</a>
        </div>
    </body>
    </html>
    """.format(img_base64, predicted_class, confidence)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
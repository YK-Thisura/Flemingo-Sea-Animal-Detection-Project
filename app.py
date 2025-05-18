# pip install Flask tensorflow pillow numpy 

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('best_sea_animal_model.h5')  

# Class labels (ensure this order matches your training labels)
class_labels = ['Coral', 'Fish', 'Jelly Fish', 'Lobster', 'Penguin', 'Seal', 'Sharks', 'Squid', 'Turtle']

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                    (w + min_dim) // 2, (h + min_dim) // 2))
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = np.argmax(preds)
    label = class_labels[class_index]
    confidence = np.max(preds)

    return f"{label} ({confidence*100:.2f}%)"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = predict_image(filepath)
            image_path = url_for('static', filename='uploads/' + filename)

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)

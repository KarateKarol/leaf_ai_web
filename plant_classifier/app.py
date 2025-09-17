from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("leaf_classifier.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file).resize((128,128))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        return f"🌿 To jest liść klasy: {predicted_class}"

    return '''
        <h1>Leaf Classifier</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Wyślij">
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)

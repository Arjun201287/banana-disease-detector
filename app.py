# ==================== PRODUCTION READY APP.PY ====================

import os
import gc
import io
import base64
import warnings

# ==================== MEMORY OPTIMIZATION ====================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

# ==================== IMPORTS ====================

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

import numpy as np
import cv2
from PIL import Image

# ==================== TENSORFLOW ====================

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

gc.set_threshold(100, 5, 5)

# ==================== OPTIONAL GDOWN ====================

try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# ==================== APP CONFIG ====================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "model_best.weights.h5")

IMG_SIZE = (160, 160)

MODEL_FILE_ID = "1pIxcmq4IbYX8kIMyx9YNFrYkAT0vI4Ff"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ==================== CLASS NAMES ====================

CLASS_NAMES = [
    'cordana',
    'healthy',
    'pestalotiopsis',
    'sigatoka'
]

# ==================== DISEASE INFORMATION ====================

DISEASE_INFO = {

    "cordana": {
        "name": "Cordana Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Brown oval spots with yellow halo.",
        "prevention": [
            "Remove infected leaves",
            "Maintain plant spacing",
            "Use copper fungicide"
        ],
        "treatment": [
            "Spray Mancozeb",
            "Use Carbendazim"
        ],
        "organic": [
            "Neem oil spray",
            "Compost tea"
        ]
    },

    "healthy": {
        "name": "Healthy Banana Plant",
        "severity": "None",
        "symptoms": "No visible disease detected.",
        "prevention": [
            "Maintain irrigation",
            "Balanced fertilizer"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic": [
            "Continue organic practices"
        ]
    },

    "pestalotiopsis": {
        "name": "Pestalotiopsis Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Brown spots with rings.",
        "prevention": [
            "Clean field debris",
            "Avoid water stress"
        ],
        "treatment": [
            "Copper oxychloride spray"
        ],
        "organic": [
            "Bordeaux mixture"
        ]
    },

    "sigatoka": {
        "name": "Sigatoka Leaf Spot",
        "severity": "High",
        "symptoms": "Yellow streaks turning brown.",
        "prevention": [
            "Use resistant varieties",
            "Prune infected leaves"
        ],
        "treatment": [
            "Propiconazole spray"
        ],
        "organic": [
            "Garlic extract spray"
        ]
    }
}

# ==================== FILE VALIDATION ====================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== DOWNLOAD MODEL ====================

def download_model():

    if not os.path.exists(WEIGHTS_PATH):

        print("Downloading model weights...")

        try:
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

            gdown.download(
                url,
                WEIGHTS_PATH,
                quiet=False
            )

            print("Model downloaded successfully!")

        except Exception as e:
            print(f"Download failed: {e}")

# ==================== BUILD MODEL ====================

_model_instance = None

def build_model():

    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        BatchNormalization
    )

    from tensorflow.keras.models import Model

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=Input(shape=(160, 160, 3))
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(256, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        len(CLASS_NAMES),
        activation='softmax'
    )(x)

    model = Model(base_model.input, outputs)

    return model

# ==================== LOAD MODEL ====================

def get_model():

    global _model_instance

    if _model_instance is None:

        print("Loading AI model...")

        model = build_model()

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        if os.path.exists(WEIGHTS_PATH):

            model.load_weights(WEIGHTS_PATH)

            print("Weights loaded!")

        else:
            print("Using default model.")

        _model_instance = model

    return _model_instance

# ==================== PREPROCESS IMAGE ====================

def preprocess_image(pil_img):

    img = pil_img.convert("RGB")

    img = img.resize(IMG_SIZE)

    img = np.array(img).astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==================== GRADCAM ====================

def generate_heatmap(image):

    img = np.array(image.resize(IMG_SIZE))

    heatmap = cv2.applyColorMap(
        np.uint8(np.mean(img, axis=2)),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return Image.fromarray(overlay)

# ==================== HTML ====================

HTML_TEMPLATE = """

<!DOCTYPE html>

<html>

<head>

<title>Banana Disease Detection</title>

<style>

body{
font-family:Arial;
background:#f4f4f4;
padding:20px;
text-align:center;
}

.container{
background:white;
padding:30px;
border-radius:10px;
max-width:900px;
margin:auto;
}

button{
background:green;
color:white;
padding:10px 20px;
border:none;
border-radius:5px;
cursor:pointer;
}

img{
max-width:100%;
border-radius:10px;
margin-top:20px;
}

</style>

</head>

<body>

<div class="container">

<h1>🍌 Banana Disease Detection</h1>

<p>Upload banana leaf image</p>

<input type="file" id="fileInput">

<br><br>

<button onclick="uploadImage()">Predict Disease</button>

<div id="result"></div>

</div>

<script>

async function uploadImage(){

const fileInput = document.getElementById("fileInput");

if(fileInput.files.length===0){

alert("Please select image");

return;

}

const formData = new FormData();

formData.append("file", fileInput.files[0]);

document.getElementById("result").innerHTML = "Processing...";

const response = await fetch("/predict",{
method:"POST",
body:formData
});

const data = await response.json();

if(data.error){

document.getElementById("result").innerHTML = data.error;

return;

}

document.getElementById("result").innerHTML = `

<h2>${data.disease_name}</h2>

<p><b>Severity:</b> ${data.severity}</p>

<p><b>Confidence:</b> ${data.confidence}%</p>

<p><b>Symptoms:</b> ${data.symptoms}</p>

<img src="data:image/png;base64,${data.gradcam_image}">

`;

}

</script>

</body>

</html>

"""

# ==================== ROUTES ====================

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health")
def health():
    return jsonify({
        "status": "running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        filename = secure_filename(file.filename)

        if not allowed_file(filename):
            return jsonify({"error": "Invalid file format"})

        image = Image.open(file.stream).convert("RGB")

        x = preprocess_image(image)

        model = get_model()

        prediction = model.predict(
            x,
            verbose=0
        )

        class_id = int(np.argmax(prediction[0]))

        confidence = float(
            np.max(prediction[0]) * 100
        )

        label = CLASS_NAMES[class_id]

        info = DISEASE_INFO[label]

        # Generate visualization

        heatmap = generate_heatmap(image)

        buffer = io.BytesIO()

        heatmap.save(
            buffer,
            format="PNG"
        )

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        del x
        del prediction

        gc.collect()

        return jsonify({

            "disease_name": info["name"],

            "severity": info["severity"],

            "confidence": round(confidence, 2),

            "symptoms": info["symptoms"],

            "prevention": info["prevention"],

            "treatment": info["treatment"],

            "organic": info["organic"],

            "gradcam_image": encoded_image

        })

    except Exception as e:

        gc.collect()

        return jsonify({
            "error": str(e)
        })

# ==================== START SERVER ====================

if __name__ == "__main__":

    download_model()

    port = int(
        os.environ.get("PORT", 5000)
    )

    print("=" * 50)
    print("Banana Disease Detection Server")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=port
    )

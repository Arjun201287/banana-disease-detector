# ==================== PRODUCTION APP FOR RENDER.COM ====================
import os
import sys
import io
import base64
import warnings
import gc
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# ==================== CRITICAL: TensorFlow Memory Optimization ====================
# MUST be set BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["TF_NUM_INTEROP_THREADS"] = "1"  # Limit inter-op threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # Limit intra-op threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

# Now import tensorflow with optimized settings
import tensorflow as tf

# Disable GPU completely (we're on CPU-only environment)
tf.config.set_visible_devices([], 'GPU')

# Limit CPU thread usage to reduce memory footprint
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Force TensorFlow to use minimal memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
            )
    except RuntimeError:
        pass

# Enable garbage collection more frequently
gc.set_threshold(100, 5, 5)

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "model_best.weights.h5")
IMG_SIZE = (224, 224)  # Smaller is better for memory
BATCH_SIZE = 1  # Always 1 for prediction

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== DOWNLOAD MODEL ====================
MODEL_FILE_ID = "1pIxcmq4IbYX8kIMyx9YNFrYkAT0vI4Ff"

def download_model_from_drive():
    """Download model weights from Google Drive if not exists"""
    if not os.path.exists(WEIGHTS_PATH):
        print("📥 Downloading model weights from Google Drive...")
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, WEIGHTS_PATH, quiet=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download from Drive: {e}")
    else:
        print("✅ Model weights found locally!")

# ==================== CLASSES & DISEASE INFO ====================
CLASS_NAMES = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']
NUM_CLASSES = len(CLASS_NAMES)

DISEASE_INFO = {
    "cordana": {
        "name": "Cordana Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Small, brown, oval spots with yellow halos. Spots enlarge and become gray centers.",
        "prevention": [
            "Remove and destroy infected leaves immediately",
            "Ensure proper plant spacing (3m x 3m) for air circulation",
            "Apply copper-based fungicides every 14 days",
            "Avoid overhead irrigation"
        ],
        "treatment": [
            "Spray Mancozeb 80% WP @ 2g/L water",
            "Apply Carbendazim @ 1g/L water if severe",
            "Remove severely infected plants"
        ],
        "organic": [
            "Neem oil spray (5ml/L water) twice weekly",
            "Baking soda solution (1g/L water)",
            "Compost tea application"
        ]
    },
    "sigatoka": {
        "name": "Sigatoka Leaf Spot (Yellow Sigatoka)",
        "severity": "High",
        "symptoms": "Light yellow streaks parallel to leaf veins, turning brown with gray centers.",
        "prevention": [
            "Use disease-resistant varieties (FHIA series)",
            "Regular pruning - remove bottom leaves",
            "Maintain proper drainage",
            "Apply protective fungicides before rainy season"
        ],
        "treatment": [
            "Propiconazole @ 1ml/L water",
            "Chlorothalonil @ 2g/L water",
            "Mix Tebuconazole + Trifloxystrobin for resistance management"
        ],
        "organic": [
            "Pseudomonas fluorescens biofungicide",
            "Garlic extract spray (10 cloves/L water)",
            "Seaweed extract for plant immunity"
        ]
    },
    "pestalotiopsis": {
        "name": "Pestalotiopsis Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Brown spots with concentric rings, black fungal fruiting bodies visible.",
        "prevention": [
            "Remove crop debris regularly",
            "Avoid water stress",
            "Apply Trichoderma biofungicide to soil",
            "Maintain field sanitation"
        ],
        "treatment": [
            "Copper oxychloride @ 2.5g/L water",
            "Carbendazim 50% WP @ 1g/L water",
            "Alternate fungicides to prevent resistance"
        ],
        "organic": [
            "Bordeaux mixture (1%) weekly",
            "Neem cake application to soil",
            "Cow urine solution (10% concentration)"
        ]
    },
    "healthy": {
        "name": "Healthy Banana Plant",
        "severity": "None",
        "symptoms": "No visible disease symptoms. Plant appears vigorous with green leaves.",
        "prevention": [
            "Regular monitoring (weekly inspection)",
            "Balanced fertilization (NPK 8-10-8)",
            "Proper irrigation management",
            "Mulching to retain moisture"
        ],
        "treatment": ["No treatment needed - maintain good agricultural practices"],
        "organic": ["Continue regular organic farming practices"]
    }
}

# ==================== LAZY MODEL LOADING ====================
# Global variable for model - loaded only on first request
_model_instance = None

def build_model(input_shape=(224,224,3), num_classes=NUM_CLASSES):
    """Build the model architecture"""
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras import Model
    
    base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    base.trainable = False
    x = GlobalMaxPooling2D()(base.output)
    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, outputs)
    return model

def get_model():
    """Lazy load model - only loads when first needed"""
    global _model_instance
    if _model_instance is None:
        print("🔄 Loading model (first request)...")
        try:
            _model_instance = build_model()
            _model_instance.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                                    loss='categorical_crossentropy', 
                                    metrics=['accuracy'])
            
            if os.path.exists(WEIGHTS_PATH):
                _model_instance.load_weights(WEIGHTS_PATH)
                print("✅ Model loaded successfully with weights!")
            else:
                print("⚠️ No weights found - using random weights (demo mode)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _model_instance = build_model()
            print("⚠️ Using fresh model as fallback")
    return _model_instance

# Download weights at startup (doesn't load model yet)
download_model_from_drive()

# ==================== PREPROCESS ====================
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(pil_img):
    """Preprocess image for model input"""
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ==================== SIMPLIFIED GRAD-CAM (Memory Efficient) ====================
def find_last_conv_layer(m):
    """Find last conv layer name"""
    for layer in reversed(m.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            return None
        
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        
        # Clean up
        del grad_model
        gc.collect()
        
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error (non-critical): {e}")
        return None

def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    if heatmap is None:
        return pil_img
    img = np.array(pil_img.convert("RGB").resize(IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)

# ==================== HTML TEMPLATE (Same as before) ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Banana Disease Detection - Farmer's Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2d5016, #4a7c23);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.95; }
        .content { padding: 40px; }
        .upload-area {
            border: 3px dashed #4a7c23;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: #f9fef5;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
        }
        .upload-area:hover { background: #f0f9e8; border-color: #2d5016; }
        .upload-area.drag-over { background: #e8f5e1; border-color: #1a4d00; }
        .upload-icon { font-size: 48px; margin-bottom: 15px; }
        .file-input { display: none; }
        .btn {
            background: #4a7c23;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
            margin: 10px;
        }
        .btn:hover { background: #2d5016; transform: translateY(-2px); }
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            display: none;
        }
        .result-card.show { display: block; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .prediction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .disease-name { font-size: 1.8em; font-weight: bold; }
        .severity {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .severity.High { background: #ff4757; color: white; }
        .severity.Moderate { background: #ffa502; color: white; }
        .severity.None { background: #2ed573; color: white; }
        .confidence { font-size: 1.2em; color: #666; margin-bottom: 20px; }
        .image-container { text-align: center; margin: 20px 0; }
        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .info-section {
            margin-top: 25px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .info-section h3 { color: #4a7c23; margin-bottom: 15px; }
        .symptoms {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }
        .prevention, .treatment, .organic { margin: 15px 0; padding: 15px; border-radius: 8px; }
        .prevention { background: #d4edda; border-left: 4px solid #28a745; }
        .treatment { background: #f8d7da; border-left: 4px solid #dc3545; }
        .organic { background: #d1ecf1; border-left: 4px solid #17a2b8; }
        .loader {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4a7c23;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .footer {
            background: #2d5016;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }
        .warning-banner {
            background: #fff3cd;
            color: #856404;
            padding: 10px;
            text-align: center;
            font-size: 0.9em;
            border-bottom: 1px solid #ffeeba;
        }
        @media (max-width: 768px) {
            .content { padding: 20px; }
            .header h1 { font-size: 1.5em; }
            .disease-name { font-size: 1.3em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍌 Banana Disease Detection Assistant</h1>
            <p>AI-Powered Diagnosis for Healthy Banana Plants</p>
        </div>
        <div class="warning-banner">
            ⚡ First prediction may take 30-60 seconds as the AI model loads. Please wait!
        </div>
        <div class="content">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <p>Click or drag & drop a banana leaf photo here</p>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">Supports: JPG, PNG, JPEG</p>
                <input type="file" id="fileInput" accept="image/*" class="file-input">
                <br>
                <button class="btn" onclick="document.getElementById('fileInput').click()">Choose Image</button>
            </div>
            
            <div class="loader" id="loader"></div>
            <div id="loadingStatus" style="text-align: center; color: #666; display: none;">Processing image...</div>
            
            <div class="result-card" id="resultCard">
                <div class="prediction-header">
                    <span class="disease-name" id="diseaseName">-</span>
                    <span class="severity" id="severity">-</span>
                </div>
                <div class="confidence" id="confidence">Confidence: -</div>
                <div class="image-container" id="imageContainer"></div>
                
                <div class="info-section">
                    <h3>📋 Symptoms</h3>
                    <div class="symptoms" id="symptoms">-</div>
                    
                    <h3>🛡️ Prevention</h3>
                    <div class="prevention" id="prevention">-</div>
                    
                    <h3>💊 Treatment</h3>
                    <div class="treatment" id="treatment">-</div>
                    
                    <h3>🌱 Organic Solutions</h3>
                    <div class="organic" id="organic">-</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🌾 Smart Farming Solution | AI for Agriculture | Instant Disease Detection</p>
            <p style="font-size: 0.8em; margin-top: 10px;">Powered by TensorFlow & EfficientNet</p>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loader = document.getElementById('loader');
        const loadingStatus = document.getElementById('loadingStatus');
        const resultCard = document.getElementById('resultCard');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImageUpload(file);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleImageUpload(e.target.files[0]);
            }
        });
        
        async function handleImageUpload(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            loader.style.display = 'block';
            loadingStatus.style.display = 'block';
            resultCard.classList.remove('show');
            loadingStatus.textContent = '📤 Uploading and analyzing image...';
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                loadingStatus.textContent = `✅ Analysis complete in ${elapsed} seconds`;
                displayResult(data);
            } catch (error) {
                alert('Error processing image. Please try again.');
                console.error(error);
                loadingStatus.textContent = '❌ Error occurred. Please try again.';
            } finally {
                loader.style.display = 'none';
                setTimeout(() => {
                    loadingStatus.style.display = 'none';
                }, 2000);
            }
        }
        
        function displayResult(data) {
            document.getElementById('diseaseName').textContent = data.disease_name;
            document.getElementById('severity').textContent = data.severity;
            document.getElementById('severity').className = `severity ${data.severity}`;
            document.getElementById('confidence').textContent = `Confidence: ${data.confidence}%`;
            document.getElementById('symptoms').innerHTML = data.symptoms;
            document.getElementById('prevention').innerHTML = data.prevention.map(p => `• ${p}`).join('<br>');
            document.getElementById('treatment').innerHTML = data.treatment.map(t => `• ${t}`).join('<br>');
            document.getElementById('organic').innerHTML = data.organic.map(o => `• ${o}`).join('<br>');
            
            if (data.gradcam_image) {
                document.getElementById('imageContainer').innerHTML = `
                    <img src="data:image/png;base64,${data.gradcam_image}" class="result-image" alt="Disease detection heatmap">
                    <p style="font-size: 0.85em; color: #666; margin-top: 10px;">🔍 Heatmap shows affected areas</p>
                `;
            }
            
            resultCard.classList.add('show');
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    </script>
</body>
</html>
'''

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": _model_instance is not None,
        "memory_optimized": True
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    """Main prediction endpoint - memory optimized"""
    result = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        pil_img = Image.open(file.stream).convert("RGB")
        
        # Preprocess image
        x = preprocess_image(pil_img)
        
        # Get model (lazy loads on first request)
        model = get_model()
        
        # Make prediction
        preds = model.predict(x, verbose=0)  # verbose=0 reduces memory
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0])) * 100
        label = CLASS_NAMES[class_id]
        
        # Get disease information
        info = DISEASE_INFO.get(label, DISEASE_INFO["healthy"])
        
        # Generate Grad-CAM heatmap (optional, skip if memory critical)
        cam_b64 = None
        try:
            heatmap = make_gradcam_heatmap(x, model, pred_index=class_id)
            if heatmap is not None:
                cam_img = overlay_heatmap(pil_img, heatmap, alpha=0.45)
                buf = io.BytesIO()
                cam_img.save(buf, format="PNG", optimize=True)  # optimize reduces size
                cam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                del cam_img
                buf.close()
        except Exception as e:
            print(f"Grad-CAM skipped (non-critical): {e}")
        
        # Prepare response
        response = {
            "disease_name": info["name"],
            "severity": info["severity"],
            "confidence": round(confidence, 2),
            "symptoms": info["symptoms"],
            "prevention": info["prevention"],
            "treatment": info["treatment"],
            "organic": info["organic"],
            "gradcam_image": cam_b64
        }
        
        # Force garbage collection after prediction
        del preds
        del x
        gc.collect()
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        gc.collect()
        return jsonify({"error": str(e)}), 500

# ==================== RUN APP ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("🍌 Banana Disease Detection API")
    print(f"📍 Running on port: {port}")
    print("⚡ Model will load on first prediction request")
    print("💾 Memory optimized for Render free tier")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port)

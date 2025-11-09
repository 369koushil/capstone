from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import tensorflow as tf
from moviepy.editor import VideoFileClip
import os
import traceback
import tempfile
from werkzeug.utils import secure_filename
from keras import layers, models 
# Import the serialization decorator (included for best practice)
from keras.saving import register_keras_serializable 

# =========================================================
# CONFIGURATION - MUST MATCH KAGGLE TRAINING SCRIPT
# =========================================================
SR = 16000          # Sample rate from Kaggle training
N_MELS = 64         # Number of mel bands from Kaggle training
MAX_PAD_LEN = 188   # Fixed frame length from Kaggle training (FIX_FRAMES)

# Define allowed file extensions and max size
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 50 * 1024 * 1024 # 50MB

# =========================================================
# CUSTOM LAYERS (CRITICAL FOR MODEL DESERIALIZATION)
# =========================================================
@register_keras_serializable() 
class TemporalAttention(layers.Layer):
    """Custom Keras Temporal Attention Layer, copied from training script."""
    def __init__(self, u=128, **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.W = layers.Dense(u, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)
        
    def call(self, H):
        score = self.v(tf.nn.tanh(self.W(H)))
        a = tf.nn.softmax(score, axis=1)
        ctx = tf.reduce_sum(a * H, axis=1)
        return ctx
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "u": self.u,
        })
        return config

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================
try:
    print("Loading models and artifacts...")
    
    # FIX: Use custom_objects to explicitly resolve the custom layer
    model = tf.keras.models.load_model(
        "ravdess_emotion.keras", 
        safe_mode=False,
        custom_objects={'TemporalAttention': TemporalAttention} 
    )
    
    scaler = joblib.load("feature_scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ Models loaded successfully!")
    print(f"  Model Input Shape: {model.input_shape}")
    print(f"  Emotions: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ ERROR loading models: Ensure ravdess_emotion.keras, feature_scaler.pkl, and label_encoder.pkl are in the same directory.")
    print(f"Error details: {e}")
    raise 

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# =========================================================
# FEATURE EXTRACTION FUNCTION 
# =========================================================
def extract_logmel(file_path, sr=SR): 
    """Extracts and prepares log-mel features matching model input size."""
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be loaded")
        
        audio = librosa.util.normalize(audio)
        
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=N_MELS, 
            n_fft=2048, 
            hop_length=512
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        
        if logmel.shape[1] > MAX_PAD_LEN:
            start = (logmel.shape[1] - MAX_PAD_LEN) // 2
            logmel = logmel[:, start:start + MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - logmel.shape[1]
            logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
            
        return logmel.T.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error during feature extraction: {str(e)}")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS

def is_video_file(filename):
    """Check if file is a video"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS

# =========================================================
# API ENDPOINT
# =========================================================
@app.route("/predict", methods=["POST"])
def predict_emotion():
    temp_files = [] 
    
    try:
        print("=== Request received ===")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded. Please include a file in the 'file' field.", "success": False}), 400
        
        file = request.files["file"]
        
        if file.filename == '':
            return jsonify({"error": "No file selected", "success": False}), 400
            
        if not allowed_file(file.filename):
             return jsonify({
                "error": "File type not supported. Allowed types: audio/video.",
                "success": False
            }), 400
        
        print(f"File received: {file.filename}")

        # 1. Save File
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
            filepath = tmp.name
            file.save(filepath)
            temp_files.append(filepath)
        
        print(f"File saved to: {filepath}")

        # 2. Extract audio from video if needed
        if is_video_file(filename):
            print("Extracting audio from video...")
            video = VideoFileClip(filepath)
            
            if video.audio is None:
                video.close()
                raise ValueError("Video file has no audio track")
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                audio_path = tmp_audio.name
                temp_files.append(audio_path)
            
            video.audio.write_audiofile(audio_path, logger=None, verbose=False)
            video.close()
            filepath = audio_path
            print(f"Audio extracted to: {filepath}")

        # 3. Extract features
        print("Extracting features...")
        features = extract_logmel(filepath)
        
        # 4. Prepare input for model: Add batch dimension (1) and channel dimension (1) if required
        
        # The raw feature shape is (188, 64)
        if len(model.input_shape) == 4 and model.input_shape[-1] == 1:
            # Final Keras input shape: (1, 188, 64, 1)
            X = features[np.newaxis, ..., np.newaxis] 
        else: 
            # Final Keras input shape: (1, 188, 64)
            X = features[np.newaxis, ...] 
        
        print(f"X input shape: {X.shape}")

        # 5. Scale features (FIXED LOGIC)
        print("Scaling features...")
        original_shape = X.shape # e.g., (1, 188, 64, 1)
        
        # CORRECT RESHAPING: Flatten the whole feature vector into one row (1, 12032)
        X_flat = X.reshape(1, -1) # Shape: (1, 12032) - matches scaler training input
        
        # Scale the flattened data
        X_flat_scaled = scaler.transform(X_flat)
        
        # Reshape back to the original Keras input shape for prediction
        X_scaled = X_flat_scaled.reshape(original_shape)

        # 6. Predict
        print("Predicting...")
        preds = model.predict(X_scaled, verbose=0)
        
        # 7. Format Output
        pred_class = np.argmax(preds, axis=1)[0]
        emotion = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(preds[0][pred_class])
        
        emotion_probs = {label_encoder.classes_[i]: float(preds[0][i]) for i in range(len(label_encoder.classes_))}
        
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": emotion_probs,
            "success": True
        })
    
    except ValueError as e:
        error_msg = str(e)
        print(f"Validation ERROR: {error_msg}")
        return jsonify({
            "error": error_msg,
            "success": False
        }), 400
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {error_msg}", "success": False}), 500
    
    finally:
        # 8. Cleanup
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up: {temp_file}")
            except Exception as e:
                print(f"Failed to clean up {temp_file}: {e}")

@app.errorhandler(413)
def file_too_large(e):
    """Handle file size errors"""
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB",
        "success": False
    }), 413

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 EMOTION RECOGNITION API STARTING")
    print(f"   Model Config: SR={SR}, Mels={N_MELS}, Frames={MAX_PAD_LEN}")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
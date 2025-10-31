from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import tensorflow as tf
from moviepy.editor import VideoFileClip
import os
import traceback

# Load model
model = tf.keras.models.load_model("ravdess_emotion.keras", safe_mode=False)
scaler = joblib.load("feature_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

N_MELS = 128
MAX_PAD_LEN = 300

app = Flask(__name__)
CORS(app)

def extract_logmel(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr)
    audio = librosa.util.normalize(audio)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512)
    logmel = librosa.power_to_db(mel, ref=np.max)
    if logmel.shape[1] > MAX_PAD_LEN:
        start = (logmel.shape[1] - MAX_PAD_LEN) // 2
        logmel = logmel[:, start:start + MAX_PAD_LEN]
    else:
        pad_width = MAX_PAD_LEN - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
    return logmel.T.astype(np.float32)

@app.route("/predict", methods=["POST"])
def predict_emotion():
    try:
        print("=== Request received ===")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "success": False}), 400
        
        file = request.files["file"]
        
        if file.filename == '':
            return jsonify({"error": "Empty filename", "success": False}), 400
        
        print(f"File received: {file.filename}")

        filepath = "temp." + file.filename.split(".")[-1]
        file.save(filepath)
        print(f"File saved to: {filepath}")

        if filepath.endswith((".mp4", ".avi", ".mov")):
            print("Extracting audio from video...")
            video = VideoFileClip(filepath)
            audio_path = "temp_audio.wav"
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            os.remove(filepath)
            filepath = audio_path
            print(f"Audio extracted to: {filepath}")

        print("Extracting features...")
        features = extract_logmel(filepath)
        print(f"Features shape: {features.shape}")
        
        X = features.reshape(1, features.shape[0], features.shape[1])
        print(f"X shape: {X.shape}")

        print("Scaling features...")
        X_2d = X.reshape(-1, X.shape[-1])
        X_2d = scaler.transform(X_2d)
        X_scaled = X_2d.reshape(X.shape)
        print(f"X_scaled shape: {X_scaled.shape}")

        print("Predicting...")
        preds = model.predict(X_scaled, verbose=0)
        
        # NEW: Print all probabilities
        print(f"Prediction probabilities:")
        for i, emotion in enumerate(label_encoder.classes_):
            print(f"  {emotion}: {preds[0][i]:.4f}")
        
        pred_class = np.argmax(preds, axis=1)[0]
        emotion = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(preds[0][pred_class])
        
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")

        if os.path.exists(filepath):
            os.remove(filepath)
            print("Temp file cleaned up")

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {label_encoder.classes_[i]: float(preds[0][i]) for i in range(len(label_encoder.classes_))},
            "success": True
        })
    
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        return jsonify({"error": error_msg, "success": False}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
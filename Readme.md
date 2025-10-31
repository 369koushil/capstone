# 🎭 Speech Emotion Detection System

AI-powered web application that detects emotions from audio and video files using deep learning.

## 📋 Overview

This project uses a Convolutional Neural Network (CNN) trained on the RAVDESS dataset to classify emotions from speech. It consists of a Flask backend API and a modern React frontend interface.

**Supported Emotions:**
- 😐 Neutral
- 😌 Calm
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

## 🏗️ Project Structure

```
speech-emotion-detection/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── ravdess_emotion.keras     # Trained model
│   ├── feature_scaler.pkl        # Feature scaler
│   ├── label_encoder.pkl         # Label encoder
│   └── requirements.txt          # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── App.jsx               # Main React component
    │   ├── App.css               # Styles
    │   └── main.jsx              # Entry point
    ├── public/                   # Static assets
    ├── package.json              # Node dependencies
    └── vite.config.js            # Vite configuration
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**

### Backend Setup

1. **Navigate to backend folder:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask flask-cors librosa numpy tensorflow joblib moviepy
   ```

4. **Required files:**
   - `ravdess_emotion.keras` - Trained model file
   - `feature_scaler.pkl` - Scaler for feature normalization
   - `label_encoder.pkl` - Label encoder for emotions

5. **Start the server:**
   ```bash
   python app.py
   ```

   Server will run on `http://127.0.0.1:5000`

### Frontend Setup

1. **Navigate to frontend folder:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   App will run on `http://localhost:5173`

## 💻 Usage

1. **Start both servers** (backend and frontend)
2. **Open browser** and go to `http://localhost:5173`
3. **Upload an audio/video file** by:
   - Clicking the upload area
   - Dragging and dropping a file
4. **Click "Detect Emotion"**
5. **View the predicted emotion** with visual feedback

### Supported File Formats

**Audio:**
- WAV
- MP3
- FLAC
- OGG

**Video:**
- MP4
- AVI
- MOV

## 🔧 API Endpoints

### POST `/predict`

Predicts emotion from uploaded audio/video file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio/video file)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.8523,
  "probabilities": {
    "neutral": 0.0234,
    "calm": 0.0156,
    "happy": 0.8523,
    "sad": 0.0089,
    "angry": 0.0234,
    "fearful": 0.0123,
    "disgust": 0.0345,
    "surprised": 0.0296
  },
  "success": true
}
```

## 🎨 Features

### Frontend
- ✨ Modern, animated UI with gradient effects
- 📱 Fully responsive design
- 🎯 Drag-and-drop file upload
- 🎭 Emotion visualization with emojis and colors
- ⚡ Real-time prediction feedback
- 🔄 Loading states and error handling

### Backend
- 🧠 Deep learning model (CNN with attention mechanism)
- 🎵 Audio feature extraction using Mel-spectrograms
- 🎬 Automatic audio extraction from videos
- 📊 Prediction confidence scores
- 🔒 CORS enabled for web access

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning
- **Librosa** - Audio processing
- **NumPy** - Numerical operations
- **MoviePy** - Video processing

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Lucide React** - Icon library
- **CSS3** - Styling with animations

## 📊 Model Details

- **Architecture:** CNN with attention mechanism
- **Input:** Log Mel-spectrograms (128 x 300)
- **Training Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Classes:** 8 emotions
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy with label smoothing

## 🐛 Troubleshooting

### Backend Issues

**Model not loading:**
```bash
# Make sure all model files are in the backend folder:
# - ravdess_emotion.keras
# - feature_scaler.pkl  
# - label_encoder.pkl
```

**Port already in use:**
```python
# Change port in app.py:
app.run(host="127.0.0.1", port=5001, debug=True)
```

### Frontend Issues

**CORS errors:**
- Ensure Flask backend has `flask-cors` installed
- Check that backend is running on port 5000

**Module not found:**
```bash
cd frontend
npm install
```

## 📝 Development

### Backend Dependencies

```txt
flask>=2.3.0
flask-cors>=4.0.0
librosa>=0.10.0
numpy>=1.24.0
tensorflow>=2.13.0
joblib>=1.3.0
moviepy>=1.0.3
```

### Frontend Dependencies

```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "lucide-react": "^0.263.1",
  "vite": "^7.1.7"
}
```

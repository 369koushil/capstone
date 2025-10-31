import { useState, useRef } from 'react';
import { Upload, Mic, Video, Loader2, Sparkles, AlertCircle } from 'lucide-react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const emotionEmojis = {
    neutral: '😐',
    calm: '😌',
    happy: '😊',
    sad: '😢',
    angry: '😠',
    fearful: '😨',
    disgust: '🤢',
    surprised: '😲'
  };

  const emotionColors = {
    neutral: '#94a3b8',
    calm: '#60a5fa',
    happy: '#fbbf24',
    sad: '#3b82f6',
    angry: '#ef4444',
    fearful: '#a855f7',
    disgust: '#10b981',
    surprised: '#f97316'
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      console.log('Sending request...');
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      console.log('Response received:', response);
      const data = await response.json();
      console.log('Data:', data);

      if (data.success) {
        setResult(data.emotion);
        console.log('Result set to:', data.emotion);
      } else {
        setError(data.error || 'Failed to detect emotion');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Emotion Detector</h1>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => setSelectedFile(e.target.files[0])}
        />
        
        {selectedFile && (
          <div>
            <p>Selected: {selectedFile.name}</p>
            <button onClick={handleUpload} disabled={loading}>
              {loading ? 'Analyzing...' : 'Detect Emotion'}
            </button>
          </div>
        )}

        {error && <div style={{color: 'red'}}>{error}</div>}

        {result && (
          <div style={{marginTop: '20px', padding: '20px', border: '2px solid green'}}>
            <h2>Detected Emotion: {result}</h2>
            <div style={{fontSize: '4rem'}}>{emotionEmojis[result]}</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
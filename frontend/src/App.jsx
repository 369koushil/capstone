import React, { useState, useRef, useCallback } from 'react';
import { Upload, Mic, Video, Loader2, Sparkles, AlertCircle, Trash2 } from 'lucide-react';

const emotionEmojis = {
  neutral: { emoji: '😐', color: '#94a3b8', text: 'Neutral' },
  calm: { emoji: '😌', color: '#60a5fa', text: 'Calm' },
  happy: { emoji: '😊', color: '#fbbf24', text: 'Happy' },
  sad: { emoji: '😢', color: '#3b82f6', text: 'Sad' },
  angry: { emoji: '😠', color: '#ef4444', text: 'Angry' },
  fearful: { emoji: '😨', color: '#a855f7', text: 'Fearful' },
  disgust: { emoji: '🤢', color: '#10b981', text: 'Disgust' },
  surprised: { emoji: '😲', color: '#f97316', text: 'Surprised' }
};

const formatFileName = (name) => {
    if (name.length > 30) {
        return name.substring(0, 15) + '...' + name.substring(name.length - 10);
    }
    return name;
};

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const clearState = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const handleFileChange = (file) => {
    if (file) {
      const allowedTypes = ['audio', 'video'];
      const fileType = file.type.split('/')[0];
      if (!allowedTypes.includes(fileType)) {
        setError('Only audio or video files are supported.');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult({
          emotion: data.emotion,
          confidence: data.confidence,
          probabilities: data.probabilities
        });
      } else {
        setError(data.error || 'Failed to detect emotion.');
      }
    } catch (err) {
      setError('Failed to connect to the backend server (Is http://127.0.0.1:5000 running?).');
      console.error('Fetch Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const FileIcon = ({ fileName }) => {
    const ext = fileName.split('.').pop().toLowerCase();
    if (['mp3', 'wav', 'ogg', 'flac', 'm4a'].includes(ext)) {
      return <Mic className="file-icon text-indigo" />;
    }
    if (['mp4', 'avi', 'mov', 'mkv'].includes(ext)) {
      return <Video className="file-icon text-red" />;
    }
    return <Upload className="file-icon text-gray" />;
  };

  const renderProbabilityBars = () => {
    if (!result || !result.probabilities) return null;

    const sortedProbs = Object.entries(result.probabilities)
      .sort(([, probA], [, probB]) => probB - probA);

    return (
      <div className="prob-container">
        <h3 className="prob-title">Probability Breakdown:</h3>
        {sortedProbs.map(([emotion, probability]) => {
          const emotionKey = emotion.toLowerCase();
          const meta = emotionEmojis[emotionKey] || { color: '#cccccc', text: emotion };
          const width = `${(probability * 100).toFixed(1)}%`;

          return (
            <div key={emotion} className="prob-bar-item">
              <span className="prob-label">{meta.text}</span>
              <div className="prob-bar-background">
                <div 
                  style={{ width, backgroundColor: meta.color }} 
                  className="prob-bar-fill"
                ></div>
              </div>
              <span className="prob-percentage">{width}</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <>
    <style>
      {`
        /* --- Base Layout --- */
        .app-container {
            min-height: 100vh;
            background-color: #f9fafb; /* bg-gray-50 */
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            font-family: 'Inter', sans-serif;
        }
        .main-card {
            width: 100%;
            max-width: 36rem;
            background-color: white;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border-radius: 0.75rem;
            padding: 2rem;
            border: 1px solid #f3ff6;
        }

        /* --- Header --- */
        .header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem; /* space-x-3 */
            margin-bottom: 2rem;
        }
        .header-icon {
            height: 2rem;
            width: 2rem;
            color: #9333ea; /* text-purple-600 */
        }
        .app-title {
            font-size: 1.875rem; /* text-3xl */
            font-weight: 800; /* font-extrabold */
            color: #111827; /* text-gray-900 */
            letter-spacing: -0.025em; /* tracking-tight */
        }

        /* --- Drop Zone --- */
        .drop-zone {
            border: 2px dashed #d1d5db; /* border-gray-300 */
            background-color: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.2s;
            position: relative;
        }
        .drag-active {
            border-color: #a855f7; /* border-purple-500 */
            background-color: #f3f3ff; /* bg-purple-50 */
        }
        .drop-zone-icon {
            margin: 0 auto;
            height: 2.5rem;
            width: 2.5rem;
            color: #9ca3af; /* text-gray-400 */
            margin-bottom: 0.5rem;
        }
        .drop-zone p {
            font-size: 0.875rem;
            color: #4b5563;
            margin: 0;
        }
        .drop-zone p:last-child {
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.25rem;
        }
        .hidden-input {
            display: none;
        }
        .absolute-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }

        /* --- Selected File --- */
        .selected-file-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background-color: #f9fafb; /* bg-gray-50 */
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-inner */
        }
        .file-info {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .file-icon {
            height: 1.25rem;
            width: 1.25rem;
        }
        .text-indigo { color: #818cf8; }
        .text-red { color: #f87171; }
        .text-gray { color: #9ca3af; }
        .file-name {
            font-size: 0.875rem;
            font-weight: 500;
            color: #374151;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px; /* Adjust max-width for small screens */
        }
        .remove-button {
            padding: 0.25rem;
            border-radius: 9999px; /* rounded-full */
            color: #9ca3af; /* text-gray-400 */
            transition: color 0.15s;
        }
        .remove-button:hover {
            color: #ef4444; /* hover:text-red-500 */
        }
        .remove-button svg {
            height: 1rem;
            width: 1rem;
        }

        /* --- Action Button --- */
        .action-area {
            margin-top: 2rem;
        }
        .action-button {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }
        .action-button:enabled {
            background-color: #7c3aed; /* bg-purple-600 */
            color: white;
        }
        .action-button:enabled:hover {
            background-color: #6d28d9; /* hover:bg-purple-700 */
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        }
        .action-button:disabled {
            background-color: #d1d5db; /* bg-gray-300 */
            color: #6b7280; /* text-gray-500 */
            cursor: not-allowed;
        }
        .action-button svg {
            height: 1.25rem;
            width: 1.25rem;
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* --- Error Display --- */
        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background-color: #fee2e2; /* bg-red-100 */
            border: 1px solid #f87171; /* border-red-400 */
            color: #b91c1c; /* text-red-700 */
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .error-box svg {
            height: 1.25rem;
            width: 1.25rem;
            flex-shrink: 0;
        }
        .error-message {
            font-size: 0.875rem;
            font-weight: 500;
        }

        /* --- Result Display --- */
        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background-color: white;
            border: 1px solid #e5e7eb; /* border-gray-200 */
            border-radius: 0.75rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        }
        .result-header {
            font-size: 1.125rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .result-header svg {
            height: 1.25rem;
            width: 1.25rem;
            color: #9333ea;
        }
        .primary-emotion-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem;
            background-color: #f5f3ff; /* bg-purple-50 */
            border-radius: 0.5rem;
            border-left: 4px solid #a855f7; /* border-l-4 border-purple-500 */
        }
        .primary-emotion-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: #4b5563;
        }
        .primary-emotion-chip {
            font-size: 1.25rem;
            font-weight: 800;
            text-transform: capitalize;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            color: white;
            display: flex;
            align-items: center;
        }
        .primary-emotion-chip span {
            margin-left: 0.5rem;
        }

        /* --- Probability Bars --- */
        .prob-container {
            margin-top: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .prob-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #374151;
        }
        .prob-bar-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .prob-label {
            width: 4rem; /* w-16 */
            font-size: 0.75rem; /* text-xs */
            color: #6b7280; /* text-gray-500 */
            text-transform: capitalize;
        }
        .prob-bar-background {
            flex: 1;
            height: 0.5rem; /* h-2 */
            border-radius: 9999px;
            overflow: hidden;
            background-color: #e5e7eb; /* bg-gray-200 */
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s;
        }
        .prob-percentage {
            width: 3rem; /* w-12 */
            text-align: right;
            font-size: 0.875rem;
            font-weight: 500;
            color: #1f2937;
        }

        /* --- Responsive Adjustments --- */
        @media (min-width: 640px) {
            .file-name {
                max-width: none;
            }
            .app-title {
                font-size: 2.25rem; /* md:text-4xl */
            }
        }
      `}
    </style>
    <div className="app-container">
      <div className="main-card">
        
        <div className="header">
          <Sparkles className="header-icon" />
          <h1 className="app-title">
            Speech Emotion Analyzer
          </h1>
        </div>

        {/* File Input/Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            onChange={(e) => handleFileChange(e.target.files[0])}
            className="hidden-input"
            id="file-upload"
          />
          
          <label 
            htmlFor="file-upload" 
            className="cursor-pointer block"
          >
            <Upload className="drop-zone-icon" />
            <p>
              Drag and drop your **audio (.wav, .mp3)** or **video (.mp4)** file here, or click to browse.
            </p>
            <p>Max size: 50MB</p>
          </label>
          
          {dragActive && (
            <div 
              className="absolute-overlay" 
            />
          )}
        </div>

        {/* Selected File & Action */}
        {selectedFile && (
          <div className="selected-file-box">
            <div className="file-info">
              <FileIcon fileName={selectedFile.name} />
              <span className="file-name">
                {formatFileName(selectedFile.name)}
              </span>
            </div>
            
            <button 
              onClick={clearState}
              className="remove-button"
              title="Remove file"
            >
              <Trash2 />
            </button>
          </div>
        )}

        <div className="action-area">
          <button 
            onClick={handleUpload} 
            disabled={!selectedFile || loading}
            className="action-button"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" />
                <span>Analyzing Audio...</span>
              </>
            ) : (
              <span>Detect Emotion</span>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-box">
            <AlertCircle />
            <span className="error-message">{error}</span>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className="result-box">
            <h2 className="result-header">
                <Sparkles className="result-header-icon" />
                <span>Analysis Complete!</span>
            </h2>
            
            <div className="primary-emotion-box">
                <span className="primary-emotion-label">Primary Emotion:</span>
                <span 
                    style={{ backgroundColor: emotionEmojis[result.emotion.toLowerCase()].color }}
                    className="primary-emotion-chip"
                >
                    {emotionEmojis[result.emotion.toLowerCase()].text}
                    <span>{emotionEmojis[result.emotion.toLowerCase()].emoji}</span>
                </span>
            </div>
            
            {renderProbabilityBars()}
          </div>
        )}

      </div>
    </div>
    </>
  );
};

export default App;
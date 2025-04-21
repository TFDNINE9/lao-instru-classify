import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import onnxruntime as ort
import json
import os
from pydantic import BaseModel
import tempfile
from typing import List, Dict, Any, Optional

# Configuration for audio processing (same as training)
class AudioConfig:
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 4  # Duration of each segment in seconds
    OVERLAP = 0.5  # 50% overlap between segments
    MAX_SEGMENTS = 5  # Maximum number of segments per audio file
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000

# Response models
class PredictionResult(BaseModel):
    instrument: str
    confidence: float
    all_probabilities: Dict[str, float]

class ModelInfo(BaseModel):
    name: str
    description: str
    supported_instruments: List[str]
    status: str

# Create FastAPI app
app = FastAPI(
    title="Lao Instrument Classifier API",
    description="API for classifying Lao musical instruments from audio files",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and metadata
onnx_session = None
class_labels = []
model_loaded = False
model_path = "models/lao_instruments_model_cnn.onnx"

# Helper functions
def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram from audio"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=AudioConfig.N_MELS,
        n_fft=AudioConfig.N_FFT,
        hop_length=AudioConfig.HOP_LENGTH,
        fmax=AudioConfig.FMAX
    )
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_audio_for_cnn(audio, sr):
    """Process audio using overlapping windows and create 2D spectrograms"""
    # Calculate segment length and hop length in samples
    segment_length = int(AudioConfig.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - AudioConfig.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, AudioConfig.MAX_SEGMENTS)
    
    # Extract spectrograms for each segment
    segment_specs = []
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        # Ensure we don't go beyond the audio length
        if end > len(audio):
            break
            
        segment = audio[start:end]
        
        # Extract mel spectrogram for this segment
        mel_spec = extract_mel_spectrogram(segment, sr)
        segment_specs.append(mel_spec)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_specs) < AudioConfig.MAX_SEGMENTS:
        # Create a zero-filled array of the same shape as other segments
        if segment_specs:
            zero_segment = np.zeros_like(segment_specs[0])
        else:
            # In case we somehow got no segments at all
            time_steps = segment_length // AudioConfig.HOP_LENGTH + 1
            zero_segment = np.zeros((AudioConfig.N_MELS, time_steps))
        
        segment_specs.append(zero_segment)
    
    # Stack segments into a single array
    # Result will have shape (MAX_SEGMENTS, n_mels, time_steps)
    return np.stack(segment_specs[:AudioConfig.MAX_SEGMENTS])

def predict_with_onnx(audio_data, sr):
    """Process audio data and make prediction using ONNX model"""
    global onnx_session, class_labels, model_loaded
    
    if not model_loaded:
        raise ValueError("Model not loaded")
    
    try:
        # Ensure sample rate matches what the model expects
        if sr != AudioConfig.SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=AudioConfig.SAMPLE_RATE)
        
        # Process audio for CNN
        specs = process_audio_for_cnn(audio_data, AudioConfig.SAMPLE_RATE)
        
        # Add batch dimension
        specs_batch = np.expand_dims(specs, axis=0)
        
        # Get input name
        input_name = onnx_session.get_inputs()[0].name
        
        # Run inference
        outputs = onnx_session.run(None, {input_name: specs_batch.astype(np.float32)})
        
        # Process results
        probabilities = outputs[0][0]  # First output, first batch item
        
        # Find the class with highest probability
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        # Create result dictionary
        result = {
            'instrument': class_labels[predicted_idx] if predicted_idx < len(class_labels) else "Unknown",
            'confidence': float(confidence),
            'all_probabilities': {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities) if i < len(class_labels)
            }
        }
        
        return result
    
    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")

# Load model at startup
@app.on_event("startup")
async def load_model():
    global onnx_session, class_labels, model_loaded
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
        
        # Load ONNX model
        onnx_session = ort.InferenceSession(model_path)
        
        # Load metadata (contains class labels)
        meta_path = model_path.replace('.onnx', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                class_labels = metadata.get('classes', [])
        else:
            # Try to load from separate labels file
            label_path = 'label_encoder.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    class_labels = [line.strip() for line in f.readlines()]
            else:
                print("Warning: No class labels found")
                class_labels = []
        
        model_loaded = True
        print(f"Model loaded successfully with {len(class_labels)} classes")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Endpoint for model info
@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    global class_labels, model_loaded
    
    if not model_loaded:
        return ModelInfo(
            name="Lao Instrument Classifier",
            description="CNN model for classifying Lao musical instruments",
            supported_instruments=[],
            status="Not loaded"
        )
    
    return ModelInfo(
        name="Lao Instrument Classifier",
        description="CNN model with attention for classifying Lao musical instruments",
        supported_instruments=class_labels,
        status="Ready"
    )

# Endpoint for prediction using uploaded file
@app.post("/predict", response_model=PredictionResult)
async def predict_instrument(file: UploadFile = File(...)):
    global model_loaded
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Load audio file
        audio_data, sr = librosa.load(temp_path, sr=None)
        
        # Delete temporary file
        os.remove(temp_path)
        
        # Process audio and make prediction
        result = predict_with_onnx(audio_data, sr)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
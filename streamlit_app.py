import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import onnxruntime as ort
import json
import os
import tempfile
from io import BytesIO
import soundfile as sf

# Configuration parameters (same as in training)
class Config:
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 4  # Duration of each segment in seconds
    OVERLAP = 0.5  # 50% overlap between segments
    MAX_SEGMENTS = 5  # Maximum number of segments per audio file
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000

# Set page configuration
st.set_page_config(
    page_title="Lao Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for model selection and information
with st.sidebar:
    st.title("Lao Instrument Classifier")
    st.markdown("---")
    
    st.subheader("About")
    st.markdown("""
    This application classifies Lao musical instruments based on audio recordings.
    
    Upload a WAV file or record audio to identify the instrument being played.
    
    The model can identify these instruments:
    - Khaen (ແຄນ)
    - So U (ຊໍອູ້)
    - Sing (ຊິ່ງ)
    - Pin (ພິນ)
    - Khong Wong (ຄ້ອງວົງ)
    - Ranad (ລະນາດ)
    """)
    
    st.markdown("---")
    
    # Model selection (if you have multiple models)
    model_type = st.selectbox(
        "Select Model",
        ["CNN with Attention (ONNX)"]
    )

# Helper functions for audio processing
def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram from audio"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=Config.N_MELS,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        fmax=Config.FMAX
    )
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_audio_for_cnn(audio, sr):
    """Process audio using overlapping windows and create 2D spectrograms"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, Config.MAX_SEGMENTS)
    
    # Extract spectrograms for each segment
    segment_specs = []
    segment_times = []  # Keep track of segment times for visualization
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        # Ensure we don't go beyond the audio length
        if end > len(audio):
            break
            
        segment = audio[start:end]
        segment_times.append((start/sr, end/sr))  # Store start and end times in seconds
        
        # Extract mel spectrogram for this segment
        mel_spec = extract_mel_spectrogram(segment, sr)
        segment_specs.append(mel_spec)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_specs) < Config.MAX_SEGMENTS:
        # Create a zero-filled array of the same shape as other segments
        if segment_specs:
            zero_segment = np.zeros_like(segment_specs[0])
        else:
            # In case we somehow got no segments at all
            time_steps = segment_length // Config.HOP_LENGTH + 1
            zero_segment = np.zeros((Config.N_MELS, time_steps))
        
        segment_specs.append(zero_segment)
        if len(segment_times) < Config.MAX_SEGMENTS:
            segment_times.append(None)  # No time for padded segments
    
    # Stack segments into a single array
    # Result will have shape (MAX_SEGMENTS, n_mels, time_steps)
    return np.stack(segment_specs[:Config.MAX_SEGMENTS]), segment_times[:Config.MAX_SEGMENTS]

# Load model and labels
@st.cache_resource
def load_model(model_path='models/lao_instruments_model_cnn.onnx'):
    """Load ONNX model and metadata"""
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, []
        
        # Load ONNX model
        ort_session = ort.InferenceSession(model_path)
        
        # Load metadata (contains class labels)
        meta_path = model_path.replace('.onnx', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                classes = metadata.get('classes', [])
        else:
            # Try to load from separate labels file
            label_path = 'label_encoder.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
            else:
                classes = []
        
        return ort_session, classes
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, []

# Predict function for ONNX
def predict_with_onnx(session, specs, classes):
    """Make prediction using ONNX model"""
    if session is None:
        return None
    
    try:
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: specs.astype(np.float32)})
        
        # Process results
        probabilities = outputs[0][0]  # First output, first batch item
        
        # Find the class with highest probability
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        # Create result dictionary
        result = {
            'instrument': classes[predicted_idx] if predicted_idx < len(classes) else "Unknown",
            'confidence': float(confidence),
            'all_probabilities': {
                classes[i]: float(prob) for i, prob in enumerate(probabilities) if i < len(classes)
            }
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Function to display spectrograms
def plot_spectrogram(spec, title="Mel Spectrogram"):
    """Plot and return a spectrogram figure"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spec, 
        x_axis='time', 
        y_axis='mel', 
        sr=Config.SAMPLE_RATE,
        hop_length=Config.HOP_LENGTH,
        fmax=Config.FMAX
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return fig

# Process the whole audio file and make prediction
def process_and_predict(audio_data, sr, model_session, class_labels):
    """Process audio file and make prediction"""
    # Ensure sample rate matches what the model expects
    if sr != Config.SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
        sr = Config.SAMPLE_RATE
    
    # Process audio for CNN
    specs, segment_times = process_audio_for_cnn(audio_data, sr)
    
    # Add batch dimension required by the model
    specs_batch = np.expand_dims(specs, axis=0)
    
    # Make prediction
    result = predict_with_onnx(model_session, specs_batch, class_labels)
    
    return result, specs, segment_times

# Function to display classification results
def display_results(result, specs, segment_times, audio_data, sr):
    """Display classification results and visualizations"""
    if result is None:
        st.error("Prediction failed. Please try again with a different audio file.")
        return
    
    # Display main result
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show instrument name and confidence
        st.subheader("Classification Result")
        instrument = result['instrument']
        confidence = result['confidence'] * 100
        
        # Display instrument name with confidence level
        confidence_color = 'green' if confidence > 90 else 'orange' if confidence > 70 else 'red'
        st.markdown(f"<h2 style='color: {confidence_color};'>{instrument}</h2>", unsafe_allow_html=True)
        st.markdown(f"Confidence: <b>{confidence:.1f}%</b>", unsafe_allow_html=True)
        
        # Display all probabilities as a bar chart
        st.subheader("Prediction Probabilities")
        probs = result['all_probabilities']
        
        # Sort probabilities in descending order
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Create bar chart
        labels = [item[0] for item in sorted_probs]
        values = [item[1] * 100 for item in sorted_probs]
        
        # Custom bar chart with highlight for the highest probability
        colors = ['#FF9900' if i == 0 else '#1E88E5' for i in range(len(sorted_probs))]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(labels, values, color=colors)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                    ha='left', va='center', fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Instrument Prediction Probabilities')
        
        st.pyplot(fig)
    
    with col2:
        # Show audio waveform
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots(figsize=(10, 2))
        times = np.arange(len(audio_data)) / sr
        ax.plot(times, audio_data, color='#1E88E5')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        
        # Highlight the segments used for classification if available
        if segment_times and segment_times[0] is not None:
            for i, (start, end) in enumerate(segment_times):
                if start is not None and end is not None:
                    ax.axvspan(start, end, alpha=0.2, color=f'C{i}')
        
        st.pyplot(fig)
    
    # Display spectrograms for each segment
    st.subheader("Mel Spectrograms (by segment)")
    
    # Create a grid for spectrograms
    cols = st.columns(min(3, Config.MAX_SEGMENTS))
    
    # Display each spectrogram
    for i, spec in enumerate(specs):
        if segment_times[i] is not None:  # Skip padded segments
            with cols[i % len(cols)]:
                start, end = segment_times[i]
                title = f"Segment {i+1}: {start:.1f}s - {end:.1f}s"
                fig = plot_spectrogram(spec, title)
                st.pyplot(fig)

# Main application
def main():
    # Load model and class labels
    model_session, class_labels = load_model()
    
    if model_session is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Title and description
    st.title("Lao Instrument Classifier")
    st.markdown("""
    This app uses a CNN model with attention mechanism to classify traditional Lao musical instruments.
    Upload a WAV file of a Lao instrument or record audio to identify the instrument.
    """)
    
    # File uploader
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    # Audio recording option
    st.subheader("Or Record Audio")
    audio_recording = st.audio_recorder(
        text="Click to record",
        recording_color="#FF4B4B",
        neutral_color="#6aa36f",
        stop_recording_text="Click to stop"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(uploaded_file, sr=None)
            
            with st.spinner("Processing audio..."):
                # Process audio and make prediction
                result, specs, segment_times = process_and_predict(audio_data, sr, model_session, class_labels)
                
                # Display results
                display_results(result, specs, segment_times, audio_data, sr)
        
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")
    
    # Process recorded audio
    elif audio_recording is not None:
        st.audio(audio_recording, format="audio/wav")
        
        try:
            # Create a temporary file to save the recording
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_recording)
                tmp_path = tmp_file.name
            
            # Load the recorded audio
            audio_data, sr = librosa.load(tmp_path, sr=None)
            
            with st.spinner("Processing audio..."):
                # Process audio and make prediction
                result, specs, segment_times = process_and_predict(audio_data, sr, model_session, class_labels)
                
                # Display results
                display_results(result, specs, segment_times, audio_data, sr)
            
            # Clean up temporary file
            os.remove(tmp_path)
        
        except Exception as e:
            st.error(f"Error processing the recorded audio: {str(e)}")
    
    # Display information about the instruments
    if not uploaded_file and not audio_recording:
        st.subheader("About Lao Musical Instruments")
        
        instruments_info = {
            "Khaen (ແຄນ)": "A mouth organ made of bamboo pipes, each with a metal reed. It was recognized by UNESCO as part of Laos' intangible cultural heritage in 2017.",
            "So U (ຊໍອູ້)": "A bowed string instrument with a resonator made from a coconut shell, producing a warm, melodic sound.",
            "Sing (ຊິ່ງ)": "A small cymbal-like percussion instrument used in ensembles, producing a bright, shimmering sound.",
            "Pin (ພິນ)": "A plucked string instrument with a resonator made from coconut shell, similar to a lute.",
            "Khong Wong (ຄ້ອງວົງ)": "A circular arrangement of small gongs in a wooden frame, used in ceremonial music.",
            "Ranad (ລະນາດ)": "A wooden xylophone with bamboo resonators underneath, playing an important role in Lao folk music."
        }
        
        # Display information in expandable sections
        for instrument, description in instruments_info.items():
            with st.expander(instrument):
                st.write(description)
                # Note: In a real app, you might add images here

if __name__ == "__main__":
    main()
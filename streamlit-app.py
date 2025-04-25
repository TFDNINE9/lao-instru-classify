import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import onnxruntime as ort
import json
import os
import tempfile
from io import BytesIO
import soundfile as sf
from scipy import signal

# Set page configuration
st.set_page_config(
    page_title="Lao Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model configuration and parameters
@st.cache_resource
def load_model_config(model_dir='models'):
    """Load model configuration and parameters"""
    model_path = os.path.join(model_dir, 'lao_instruments_model_optimized.onnx')
    meta_path = os.path.join(model_dir, 'lao_instruments_model_optimized_meta.json')
    preprocess_path = os.path.join(model_dir, 'lao_instruments_model_optimized_preprocess.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None
    
    if not os.path.exists(meta_path):
        st.error(f"Metadata file not found: {meta_path}")
        return None, None, None
    
    if not os.path.exists(preprocess_path):
        st.error(f"Preprocessing parameters file not found: {preprocess_path}")
        return None, None, None
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Load preprocessing parameters
    with open(preprocess_path, 'r') as f:
        preprocess_params = json.load(f)
    
    # Create ONNX inference session
    try:
        session = ort.InferenceSession(model_path)
        return session, metadata, preprocess_params
    except Exception as e:
        st.error(f"Error loading ONNX model: {str(e)}")
        return None, None, None

# Load model, metadata, and preprocessing parameters
model_session, model_metadata, preprocess_params = load_model_config()

# Extract parameters from loaded config
if preprocess_params is not None:
    SAMPLE_RATE = preprocess_params.get('sample_rate', 44100)
    SEGMENT_DURATION = preprocess_params.get('segment_duration', 3.0)
    OVERLAP = preprocess_params.get('overlap', 0.3)
    MAX_SEGMENTS = preprocess_params.get('max_segments', 3)
    N_MELS = preprocess_params.get('n_mels', 128)
    N_FFT = preprocess_params.get('n_fft', 2048)
    HOP_LENGTH = preprocess_params.get('hop_length', 512)
    FMAX = preprocess_params.get('fmax', 8000)
    APPLY_NOISE_REDUCTION = preprocess_params.get('use_noise_reduction', True)
    USE_SPECTRAL_CONTRAST = preprocess_params.get('use_spectral_contrast', True)
    USE_SPECTRAL_BANDWIDTH = preprocess_params.get('use_spectral_bandwidth', True)
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 3.0
    OVERLAP = 0.3
    MAX_SEGMENTS = 3
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000
    APPLY_NOISE_REDUCTION = True
    USE_SPECTRAL_CONTRAST = True
    USE_SPECTRAL_BANDWIDTH = True

# Get class labels from metadata
if model_metadata is not None:
    CLASS_LABELS = model_metadata.get('classes', [])
    CLASS_INDICES = model_metadata.get('class_indices', {})
else:
    # Default class labels if loading failed
    CLASS_LABELS = ['background', 'khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']
    CLASS_INDICES = {label: i for i, label in enumerate(CLASS_LABELS)}

# Instrument information and descriptions
INSTRUMENT_INFO = {
    'khean': {
        'name': 'Khaen (ແຄນ)',
        'description': 'A mouth organ made of bamboo pipes, each with a metal reed. It is considered the symbol of Lao music and was recognized by UNESCO as part of Lao\'s intangible cultural heritage in 2017.',
        'image': 'assets/khean.jpg'
    },
    'khong_vong': {
        'name': 'Khong Wong (ຄ້ອງວົງ)',
        'description': 'A circular arrangement of small gongs in a wooden frame, used in ceremonial music and traditional ensembles.',
        'image': 'assets/khong_vong.jpg'
    },
    'pin': {
        'name': 'Pin (ພິນ)',
        'description': 'A plucked string instrument with a resonator made from coconut shell, similar to a lute.',
        'image': 'assets/pin.jpg'
    },
    'ranad': {
        'name': 'Ranad (ລະນາດ)',
        'description': 'A wooden xylophone with bamboo resonators underneath, playing an important role in Lao folk music.',
        'image': 'assets/ranad.jpg'
    },
    'saw': {
        'name': 'So U (ຊໍອູ້)',
        'description': 'A bowed string instrument with a resonator made from a coconut shell, producing a warm, melodic sound.',
        'image': 'assets/saw.jpg'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'A small cymbal-like percussion instrument used in ensembles, producing a bright, shimmering sound.',
        'image': 'assets/sing.jpg'
    },
    'background': {
        'name': 'Background Noise',
        'description': 'Ambient sounds or noise that does not contain a Lao musical instrument.',
        'image': 'assets/background.jpg'
    }
}

# Adaptive noise reduction based on signal characteristics
def reduce_noise(audio, sr):
    """Apply adaptive noise reduction based on signal analysis"""
    if not APPLY_NOISE_REDUCTION:
        return audio
    
    # Analyze signal to detect noise level
    frame_length = 2048
    hop_length = 512
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find noise threshold (lowest 10% of energy)
    noise_threshold = np.percentile(rms, 10)
    noise_mask = rms <= noise_threshold * 1.2  # Allow some margin
    
    if np.any(noise_mask):
        # Extract noise segment
        noise_frames = np.where(noise_mask)[0]
        noise_segment = np.zeros(frame_length)
        count = 0
        
        for frame in noise_frames[:10]:  # Use up to 10 noise frames
            start = frame * hop_length
            end = start + frame_length
            if end <= len(audio):
                noise_segment += audio[start:end]
                count += 1
        
        if count > 0:
            noise_segment /= count  # Average noise profile
            
            # Compute noise spectrum
            noise_spec = np.abs(librosa.stft(noise_segment, n_fft=N_FFT))
            
            # Compute signal spectrum
            signal_spec = librosa.stft(audio, n_fft=N_FFT)
            signal_mag = np.abs(signal_spec)
            signal_phase = np.angle(signal_spec)
            
            # Spectral subtraction with adaptive threshold
            strength = 1.5
            spec_sub = np.maximum(
                signal_mag - strength * noise_spec.mean(axis=1, keepdims=True), 
                0.01 * signal_mag
            )
            
            # Reconstruct signal
            audio_denoised = librosa.istft(spec_sub * np.exp(1j * signal_phase), hop_length=HOP_LENGTH)
            
            # Ensure same length
            if len(audio_denoised) < len(audio):
                audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
            else:
                audio_denoised = audio_denoised[:len(audio)]
                
            return audio_denoised
    
    return audio

# Extract optimized features for classification
def extract_optimized_features(audio, sr):
    """Extract features optimized for Lao instrument classification"""
    # Apply noise reduction
    audio = reduce_noise(audio, sr)
    
    features_list = []
    
    # 1. Mel spectrogram - base features
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features_list.append(mel_spec_db)
    
    # 2. Spectral contrast - helps distinguish instruments
    if USE_SPECTRAL_CONTRAST:
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        # Resize to match mel spectrogram dimensions
        if contrast.shape[1] != mel_spec_db.shape[1]:
            contrast = signal.resample(contrast, mel_spec_db.shape[1], axis=1)
        # Normalize to similar range as mel spectrogram
        contrast = (contrast - np.mean(contrast)) / (np.std(contrast) + 1e-10) * np.std(mel_spec_db)
        features_list.append(contrast)
    
    # 3. Spectral bandwidth - useful for percussion vs wind/string distinction
    if USE_SPECTRAL_BANDWIDTH:
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        # Expand to match mel bands
        bandwidth_expanded = np.repeat(bandwidth, N_MELS // bandwidth.shape[0], axis=0)
        if bandwidth_expanded.shape[0] < N_MELS:
            bandwidth_expanded = np.pad(
                bandwidth_expanded, 
                ((0, N_MELS - bandwidth_expanded.shape[0]), (0, 0))
            )
        # Normalize
        bandwidth_expanded = (bandwidth_expanded - np.mean(bandwidth_expanded)) / (np.std(bandwidth_expanded) + 1e-10) * np.std(mel_spec_db)
        features_list.append(bandwidth_expanded)
    
    # Stack features as channels
    stacked_features = np.stack(features_list, axis=-1)
    
    return stacked_features

# Process audio with overlapping segments
def process_audio_segments(audio, sr):
    """Process audio using overlapping segments"""
    # Calculate segment length and hop length in samples
    segment_length = int(SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, MAX_SEGMENTS)
    
    # Extract features for each segment
    segment_features = []
    segment_times = []  # For visualization
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        # Ensure we don't go beyond the audio length
        if end > len(audio):
            break
            
        segment = audio[start:end]
        segment_times.append((start/sr, end/sr))
        
        # Extract optimized features
        features = extract_optimized_features(segment, sr)
        segment_features.append(features)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_features) < MAX_SEGMENTS:
        if segment_features:
            zero_segment = np.zeros_like(segment_features[0])
            segment_features.append(zero_segment)
            if len(segment_times) < MAX_SEGMENTS:
                segment_times.append(None)  # No time for padded segments
        else:
            # Should not happen with normal audio files
            st.warning("Warning: Failed to extract any segments!")
            # Create dummy feature set
            num_channels = (1 + 
                          (1 if USE_SPECTRAL_CONTRAST else 0) + 
                          (1 if USE_SPECTRAL_BANDWIDTH else 0))
            dummy_feature = np.zeros((N_MELS, segment_length // HOP_LENGTH + 1, num_channels))
            segment_features.append(dummy_feature)
            segment_times.append(None)
    
    return np.array(segment_features[:MAX_SEGMENTS]), segment_times[:MAX_SEGMENTS]

# Make prediction using ONNX model
def predict_instrument(audio_data, sr):
    """Process audio and make prediction using ONNX model"""
    if model_session is None:
        st.error("Model not loaded properly!")
        return None
    
    try:
        # Ensure audio has the right sample rate
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Process audio into segments
        features, segment_times = process_audio_segments(audio_data, sr)
        
        # Add batch dimension
        features_batch = np.expand_dims(features, axis=0).astype(np.float32)
        
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Run inference
        outputs = model_session.run(None, {input_name: features_batch})
        
        # Process results
        probabilities = outputs[0][0]  # First output, first batch item
        
        # Create result dictionary
        result = {
            'probabilities': {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'segment_times': segment_times
        }
        
        # Find the most likely instrument
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        instrument = CLASS_LABELS[max_prob_idx]
        
        # Calculate entropy as uncertainty measure
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log2(probabilities + epsilon)) / np.log2(len(probabilities))
        
        # Determine if prediction is uncertain (high entropy or low confidence)
        is_uncertain = entropy > 0.5 or max_prob < 0.5
        is_background = instrument == 'background' and max_prob > 0.5
        
        result.update({
            'instrument': instrument,
            'confidence': float(max_prob),
            'entropy': float(entropy),
            'is_uncertain': is_uncertain,
            'is_background': is_background
        })
        
        return result
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Plot waveform and spectrogram
def plot_audio_visualization(audio, sr, result=None):
    """Create visualization of the audio with classification results"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title('Waveform')
    
    # Highlight segments used for classification if available
    if result and 'segment_times' in result:
        for i, time_range in enumerate(result['segment_times']):
            if time_range is not None:
                start, end = time_range
                ax[0].axvspan(start, end, alpha=0.2, color=f'C{i}')
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title('Spectrogram')
    
    plt.tight_layout()
    return fig

# Create bar chart of classification probabilities
def plot_classification_probabilities(result):
    """Create bar chart of classification probabilities"""
    if not result or 'probabilities' not in result:
        return None
    
    # Get probabilities and sort by value
    probs = result['probabilities']
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    instruments = [INSTRUMENT_INFO.get(label, {}).get('name', label) for label, _ in sorted_probs]
    values = [prob * 100 for _, prob in sorted_probs]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create bars with highlighted top prediction
    colors = ['#FF9900' if i == 0 else '#1E88E5' for i in range(len(sorted_probs))]
    bars = ax.barh(instruments, values, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Instrument Classification Results')
    
    plt.tight_layout()
    return fig

# Display instrument information
def display_instrument_info(instrument_id):
    """Display information about a specific instrument"""
    if instrument_id not in INSTRUMENT_INFO:
        st.warning(f"Information not available for instrument: {instrument_id}")
        return
    
    info = INSTRUMENT_INFO[instrument_id]
    
    # Try to load an image if available
    try:
        if os.path.exists(info['image']):
            st.image(info['image'], caption=info['name'], width=300)
        else:
            st.write(f"## {info['name']}")
    except:
        st.write(f"## {info['name']}")
    
    st.write(info['description'])
    
    # Add more details or links if available

# Main application
def main():
    # Sidebar
    with st.sidebar:
        st.title("Lao Instrument Classifier")
        st.markdown("---")
        
        st.subheader("About")
        st.markdown("""
        This application classifies traditional Lao musical instruments based on audio recordings.
        
        Upload a WAV file or record audio directly to identify the instrument being played.
        
        This model can identify these instruments:
        """)
        
        # List instruments
        for label in [l for l in CLASS_LABELS if l != 'background']:
            info = INSTRUMENT_INFO.get(label, {})
            st.markdown(f"- **{info.get('name', label)}**")
        
        st.markdown("---")
        
        # Show technical info if requested
        if st.checkbox("Show Technical Details"):
            st.subheader("Model Information")
            st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
            st.write(f"Segment Duration: {SEGMENT_DURATION} seconds")
            st.write(f"Feature Extraction: Mel Spectrogram + Additional Features")
    
    # Main content
    st.title("🎵 Lao Instrument Classifier")
    st.markdown("""
    Upload an audio recording of a Lao musical instrument, and this app will identify which instrument is playing!
    """)
    
    # Audio input methods
    input_method = st.radio(
        "How would you like to input audio?",
        ("Upload an audio file", "Record audio")
    )
    
    if input_method == "Upload an audio file":
        # File uploader
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            # Display and process the uploaded file
            st.audio(uploaded_file, format="audio/wav")
            
            # Load and process the audio
            with st.spinner("Processing audio..."):
                try:
                    # Save to a temporary file (workaround for librosa loading)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load the audio
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)  # Clean up temp file
                    
                    # Make prediction
                    result = predict_instrument(audio_data, sr)
                    
                    if result:
                        # Show result visualization
                        st.subheader("Analysis Results")
                        
                        # Display prediction result
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if result['is_uncertain']:
                                st.warning(f"⚠️ Uncertain classification (Confidence: {result['confidence']*100:.1f}%)")
                                st.write("The model is not confident about its prediction. This might not be one of the supported Lao instruments.")
                            elif result['is_background']:
                                st.info("🔊 Background noise detected")
                                st.write("No clear Lao instrument detected in this recording.")
                            else:
                                instrument_id = result['instrument']
                                info = INSTRUMENT_INFO.get(instrument_id, {})
                                st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
                                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        
                        with col2:
                            # Plot probabilities
                            prob_fig = plot_classification_probabilities(result)
                            if prob_fig:
                                st.pyplot(prob_fig)
                        
                        # Audio visualization
                        st.subheader("Audio Visualization")
                        vis_fig = plot_audio_visualization(audio_data, sr, result)
                        st.pyplot(vis_fig)
                        
                        # Show instrument information if a specific instrument was detected
                        if not result['is_uncertain'] and not result['is_background']:
                            st.subheader("Instrument Information")
                            display_instrument_info(result['instrument'])
                    else:
                        st.error("Failed to process the audio. Please try another file.")
                
                except Exception as e:
                    st.error(f"Error processing the audio file: {str(e)}")
    
    else:  # Record audio
        # Audio recorder widget
        st.write("Click below to record audio")
        audio_recording = st.audio_recorder(
            text="Click to record",
            recording_color="#FF4B4B",
            neutral_color="#6aa36f",
            stop_recording_text="Click to stop"
        )
        
        if audio_recording is not None:
            # Display the recorded audio
            st.audio(audio_recording, format="audio/wav")
            
            # Process the recording
            with st.spinner("Processing audio..."):
                try:
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_recording)
                        tmp_path = tmp_file.name
                    
                    # Load the audio
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)  # Clean up temp file
                    
                    # Make prediction
                    result = predict_instrument(audio_data, sr)
                    
                    if result:
                        # Show result visualization
                        st.subheader("Analysis Results")
                        
                        # Display prediction result
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if result['is_uncertain']:
                                st.warning(f"⚠️ Uncertain classification (Confidence: {result['confidence']*100:.1f}%)")
                                st.write("The model is not confident about its prediction. This might not be one of the supported Lao instruments.")
                            elif result['is_background']:
                                st.info("🔊 Background noise detected")
                                st.write("No clear Lao instrument detected in this recording.")
                            else:
                                instrument_id = result['instrument']
                                info = INSTRUMENT_INFO.get(instrument_id, {})
                                st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
                                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        
                        with col2:
                            # Plot probabilities
                            prob_fig = plot_classification_probabilities(result)
                            if prob_fig:
                                st.pyplot(prob_fig)
                        
                        # Audio visualization
                        st.subheader("Audio Visualization")
                        vis_fig = plot_audio_visualization(audio_data, sr, result)
                        st.pyplot(vis_fig)
                        
                        # Show instrument information if a specific instrument was detected
                        if not result['is_uncertain'] and not result['is_background']:
                            st.subheader("Instrument Information")
                            display_instrument_info(result['instrument'])
                    else:
                        st.error("Failed to process the recorded audio. Please try recording again.")
                
                except Exception as e:
                    st.error(f"Error processing the recorded audio: {str(e)}")
    
    # Information about all instruments
    st.markdown("---")
    with st.expander("Learn About Lao Musical Instruments"):
        # Display information about each instrument
        for instrument_id in [label for label in CLASS_LABELS if label != 'background']:
            st.markdown(f"### {INSTRUMENT_INFO.get(instrument_id, {}).get('name', instrument_id)}")
            st.markdown(INSTRUMENT_INFO.get(instrument_id, {}).get('description', 'No description available'))
            st.markdown("---")

if __name__ == "__main__":
    main()
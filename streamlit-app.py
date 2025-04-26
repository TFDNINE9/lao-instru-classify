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

# Set page configuration
st.set_page_config(
    page_title="Lao Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model configuration and parameters
@st.cache_resource
def load_model_config(model_dir='models/optimized_mfcc_model'):
    """Load model configuration and parameters"""
    model_path = os.path.join(model_dir, 'optimized_mfcc_model.onnx')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    if not os.path.exists(metadata_path):
        st.error(f"Metadata file not found: {metadata_path}")
        return None, None
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create ONNX inference session
    try:
        session = ort.InferenceSession(model_path)
        return session, metadata
    except Exception as e:
        st.error(f"Error loading ONNX model: {str(e)}")
        return None, None

# Load model and metadata
model_session, model_metadata = load_model_config()

# Extract parameters from loaded config
if model_metadata is not None:
    SAMPLE_RATE = model_metadata.get('sample_rate', 44100)
    N_MFCC = model_metadata.get('n_mfcc', 40)
    N_FFT = model_metadata.get('n_fft', 2048)
    HOP_LENGTH = model_metadata.get('hop_length', 512)
    SEGMENT_DURATION = model_metadata.get('segment_duration', 3.0)
    USE_DELTA = model_metadata.get('use_delta', True)
    USE_DELTA_DELTA = model_metadata.get('use_delta_delta', True)
    CLASS_LABELS = model_metadata.get('class_names', [])
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_DURATION = 3.0
    USE_DELTA = True
    USE_DELTA_DELTA = True
    CLASS_LABELS = ['background', 'khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']

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

def extract_mfcc_features(audio, sr):
    """Extract MFCC features with deltas"""
    # Ensure audio has the desired length
    desired_length = int(SEGMENT_DURATION * sr)
    if len(audio) > desired_length:
        audio = audio[:desired_length]
    else:
        audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
    
    # Basic MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    
    features = [mfccs]
    
    # Add delta features
    if USE_DELTA:
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs = (delta_mfccs - np.mean(delta_mfccs, axis=1, keepdims=True)) / (np.std(delta_mfccs, axis=1, keepdims=True) + 1e-8)
        features.append(delta_mfccs)
    
    # Add delta-delta features
    if USE_DELTA_DELTA:
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta2_mfccs = (delta2_mfccs - np.mean(delta2_mfccs, axis=1, keepdims=True)) / (np.std(delta2_mfccs, axis=1, keepdims=True) + 1e-8)
        features.append(delta2_mfccs)
    
    # Stack all features
    combined_features = np.vstack(features)
    
    # Fix the time dimension to a consistent length
    expected_time_frames = int(np.ceil(desired_length / HOP_LENGTH)) + 1
    
    # If we got more frames than expected, truncate
    if combined_features.shape[1] > expected_time_frames:
        combined_features = combined_features[:, :expected_time_frames]
    # If we got fewer frames than expected, pad
    elif combined_features.shape[1] < expected_time_frames:
        pad_width = expected_time_frames - combined_features.shape[1]
        combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
    
    return combined_features

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
        
        # Extract MFCC features
        features = extract_mfcc_features(audio_data, sr)
        
        # Reshape for DNN (flatten features and time)
        features_flat = features.flatten()
        
        # Add batch dimension
        features_batch = np.expand_dims(features_flat, axis=0).astype(np.float32)
        
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
            }
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

def plot_audio_visualization(audio, sr, result=None):
    """Create visualization of the audio with classification results"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title('Waveform')
    
    # Plot MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1], hop_length=HOP_LENGTH)
    fig.colorbar(img, ax=ax[1])
    ax[1].set_title('MFCC Features')
    
    plt.tight_layout()
    return fig

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

def main():
    # Sidebar
    with st.sidebar:
        st.title("Lao Instrument Classifier")
        st.markdown("---")
        
        st.subheader("About")
        st.markdown("""
        This application classifies traditional Lao musical instruments based on audio recordings using a Deep Neural Network with MFCC features.
        
        Upload a WAV file to identify the instrument being played.
        
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
            st.write(f"MFCC Coefficients: {N_MFCC}")
            st.write(f"Feature Extraction: MFCC + Delta + Delta-Delta")
            st.write(f"Model Type: Deep Neural Network")
            st.write(f"Segment Duration: {SEGMENT_DURATION} seconds")
    
    # Main content
    st.title("🎵 Lao Instrument Classifier (DNN)")
    st.markdown("""
    Upload an audio recording of a Lao musical instrument, and this app will identify which instrument is playing using a Deep Neural Network!
    """)
    
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
                import traceback
                st.error(f"Full error: {traceback.format_exc()}")
    
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
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import random
import json
import seaborn as sns
import scipy.signal
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Set environment variables for Intel GPU optimization
os.environ["OPENVINO_DEVICE"] = "GPU"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Import OpenVINO if available
try:
    import openvino
    import openvino.runtime as ov
    from openvino.runtime import Core
    print(f"OpenVINO version: {openvino.__version__}")
    HAVE_OPENVINO = True
except ImportError:
    print("OpenVINO not found. Will use standard TensorFlow.")
    HAVE_OPENVINO = False

# Configuration parameters
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 5  # Duration of each segment in seconds
    OVERLAP = 0.5  # 50% overlap between segments
    MAX_SEGMENTS = 6  # Maximum number of segments per audio file
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000
    
    # Model parameters
    BATCH_SIZE = 64  # Increased for GPU
    EPOCHS = 150  # Increased max epochs
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15  # Increased patience
    
    # Data augmentation
    NUM_AUGMENTATIONS = 3  # Increased from 2
    
    # Paths
    DATA_PATH = "dataset"  # Update this to your data folder path
    MODEL_SAVE_PATH = "lao_instruments_model_cnn_improved"
    
    # Training options
    USE_CLASS_WEIGHTS = True
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_GAMMA = 2.0
    
    # Model ensemble
    USE_ENSEMBLE = True
    NUM_ENSEMBLE_MODELS = 3
    
    # OpenVINO Configuration
    USE_OPENVINO = True
    OPENVINO_DEVICE = "GPU.0"  # Use GPU.0 for first Intel GPU
    OPENVINO_PRECISION = "FP16"  # Use FP16 for faster training
    OPENVINO_BATCH_SIZE = 64  # Optimize batch size for GPU
    OPENVINO_NUM_STREAMS = 4  # For better parallelism

# Print configuration
print("Lao Instrument Classification - Improved CNN Model Training with Intel GPU")
print(f"TensorFlow version: {tf.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"Using OpenVINO: {HAVE_OPENVINO}")
print("-" * 50)

# Print configuration details
print("Configuration:")
for attr in dir(Config):
    if not attr.startswith("__"):
        print(f"  {attr}: {getattr(Config, attr)}")
print("-" * 50)

# Configure TensorFlow for Intel GPU
def configure_tensorflow_for_intel():
    """Configure TensorFlow for better performance on Intel hardware"""
    print("Configuring TensorFlow for Intel hardware...")
    
    # Set memory growth for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPUs: {gpus}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Set memory growth for {gpu}")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
    else:
        print("No standard GPUs found - will check for Intel GPU")
    
    # Performance optimization flags
    tf.config.optimizer.set_jit(True)  # Enable XLA
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    os.environ["TF_MKL_ALLOC_MAX_BYTES"] = "0"
    
    # Attempt to detect Intel GPU using OpenVINO
    if HAVE_OPENVINO:
        try:
            core = Core()
            available_devices = core.available_devices
            print(f"OpenVINO available devices: {available_devices}")
            
            if "GPU" in available_devices:
                # Set Intel GPU specific settings
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                os.environ["USE_INTEL_SDP"] = "1"  # Intel Software Development Platform
                
                print(f"Intel GPU detected through OpenVINO")
                
                # Get device info
                gpu_props = core.get_property("GPU", "FULL_DEVICE_NAME")
                if gpu_props:
                    print(f"Intel GPU details: {gpu_props}")
                
                return True
        except Exception as e:
            print(f"Error detecting Intel GPU with OpenVINO: {e}")
    
    print("Intel GPU not detected or not properly configured")
    return False

# Initialize OpenVINO runtime
def initialize_openvino():
    """Initialize OpenVINO runtime and check available devices"""
    if not HAVE_OPENVINO:
        return None
    
    try:
        # Create OpenVINO Core instance
        core = Core()
        
        # Get available devices
        available_devices = core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        # Check if GPU is available
        if "GPU" in available_devices:
            device = Config.OPENVINO_DEVICE
            print(f"Using OpenVINO with device: {device}")
            
            # Get device properties
            try:
                device_properties = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"Device details: {device_properties}")
            except Exception as e:
                print(f"Could not get device properties: {e}")
            
            return core
        else:
            print("Intel GPU not found in OpenVINO devices")
            return None
    
    except Exception as e:
        print(f"Error initializing OpenVINO: {e}")
        return None

# Utility functions
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Enhanced data augmentation functions
def enhanced_augment_audio(audio, sr, instrument_class=None):
    """Apply class-adaptive augmentations to audio sample"""
    augmented_samples = []
    
    # Determine number of augmentations based on class
    num_augmentations = Config.NUM_AUGMENTATIONS
    if instrument_class in ['ranad', 'background']:
        num_augmentations += 2  # Extra augmentations for problematic classes
    
    for i in range(num_augmentations):
        aug_audio = audio.copy()
        
        # 1. Time stretching (0.85-1.15x speed)
        if random.random() > 0.3:  # Increased probability
            stretch_factor = np.random.uniform(0.85, 1.15)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        
        # 2. Pitch shifting (up to ±3 semitones)
        if random.random() > 0.3:  # Increased probability
            pitch_shift = np.random.uniform(-3, 3)  # Increased range
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=pitch_shift)
        
        # 3. Add various types of background noise for better robustness
        if random.random() > 0.3:
            noise_type = random.choice(['white', 'pink', 'brown'])
            noise_factor = np.random.uniform(0.01, 0.08)
            
            if noise_type == 'white':
                noise = np.random.randn(len(aug_audio))
            elif noise_type == 'pink':
                noise = np.random.randn(len(aug_audio))
                # Simple approximation of pink noise
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                noise = scipy.signal.lfilter(b, a, noise)
            else:  # brown noise
                noise = np.random.randn(len(aug_audio))
                # Simple approximation of brown noise
                noise = np.cumsum(noise) / 100
                noise = noise - np.mean(noise)
                
            aug_audio = aug_audio + noise_factor * noise / np.std(noise)
        
        # 4. Time shifting (up to 30% of duration)
        if random.random() > 0.5:
            shift_factor = int(np.random.uniform(-0.3, 0.3) * len(aug_audio))  # Increased range
            if shift_factor > 0:
                aug_audio = np.pad(aug_audio, (0, shift_factor), mode='constant')[:len(audio)]
            else:
                aug_audio = np.pad(aug_audio, (abs(shift_factor), 0), mode='constant')[abs(shift_factor):]
                if len(aug_audio) < len(audio):
                    aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        # 5. Volume adjustment (0.7-1.3x)
        if random.random() > 0.3:
            volume_factor = np.random.uniform(0.7, 1.3)
            aug_audio = aug_audio * volume_factor
        
        # 6. Add specialized augmentation for percussive sounds (sing)
        if instrument_class == 'sing' and random.random() > 0.7:
            # Add reverb effect (simulates different spaces)
            reverb_level = np.random.uniform(0.1, 0.4)
            # Simple convolution reverb approximation
            impulse_response = np.exp(-np.linspace(0, 8, int(sr * 0.5)))
            impulse_response = impulse_response / np.sum(impulse_response)
            reverb = scipy.signal.convolve(aug_audio, impulse_response, mode='full')[:len(aug_audio)]
            aug_audio = (1 - reverb_level) * aug_audio + reverb_level * reverb
        
        # 7. Dynamics processing for some instruments
        if instrument_class in ['khong_vong', 'ranad', 'sing'] and random.random() > 0.7:
            # Simple dynamics processing
            threshold = np.max(np.abs(aug_audio)) * 0.3
            ratio = np.random.uniform(0.5, 2.0)  # < 1 expansion, > 1 compression
            
            mask = np.abs(aug_audio) > threshold
            aug_audio[mask] = np.sign(aug_audio[mask]) * (
                threshold + (np.abs(aug_audio[mask]) - threshold) / ratio
            )
            
        # 8. Frequency masking (for increased robustness)
        if random.random() > 0.7:
            # Apply a bandpass filter to simulate frequency masking
            low_cutoff = np.random.uniform(80, 500)
            high_cutoff = np.random.uniform(4000, 8000)
            
            # Normalized frequencies for Butterworth filter
            nyquist = sr / 2
            low = low_cutoff / nyquist
            high = high_cutoff / nyquist
            
            # Apply bandpass filter
            b, a = scipy.signal.butter(3, [low, high], btype='band')
            aug_audio = scipy.signal.filtfilt(b, a, aug_audio)
        
        # Ensure consistent length
        if len(aug_audio) > len(audio):
            aug_audio = aug_audio[:len(audio)]
        elif len(aug_audio) < len(audio):
            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

# Onset detection for percussive sounds like "sing"
def detect_percussion_events(audio, sr):
    """Detect percussion onsets and extract segments around them"""
    # Use librosa's onset detection
    onset_frames = librosa.onset.onset_detect(
        y=audio, 
        sr=sr, 
        units='frames',
        hop_length=Config.HOP_LENGTH,
        backtrack=True,
        pre_max=20,  # Increased for short percussive sounds
        post_max=20,
        pre_avg=100,
        post_avg=100,
        delta=0.2,  # More sensitive detection for percussive sounds
        wait=30
    )
    
    # Convert frames to time (in samples)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=Config.HOP_LENGTH)
    
    # Create segments around onsets
    segments = []
    segment_times = []
    
    for onset in onset_samples:
        # Create window around onset (e.g., 100ms before, 900ms after)
        pre_onset = int(0.1 * sr)  # 100ms before
        post_onset = int(0.9 * sr)  # 900ms after
        
        start = max(0, onset - pre_onset)
        end = min(len(audio), onset + post_onset)
        
        segment = audio[start:end]
        
        # Create fixed-length segment with the onset centered
        fixed_length = int(Config.SEGMENT_DURATION * sr)  # 5 second segment (from Config)
        if len(segment) < fixed_length:
            # Pad shorter segments
            padded = np.zeros(fixed_length)
            offset = (fixed_length - len(segment)) // 2
            padded[offset:offset+len(segment)] = segment
            segment = padded
        else:
            # Trim longer segments around center
            center = len(segment) // 2
            half_length = fixed_length // 2
            segment = segment[center-half_length:center+half_length]
        
        segments.append(segment)
        segment_times.append((start/sr, end/sr))
    
    return segments, segment_times

# Identify background/silence segments
def identify_background_segments(audio, sr):
    """Identify background/silence segments in audio"""
    # Calculate energy
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find low-energy segments (background)
    threshold = np.mean(rms) * 0.5  # Adjust threshold as needed
    background_frames = np.where(rms < threshold)[0]
    
    # Convert frames to samples and identify segments
    background_segments = []
    
    if len(background_frames) > 0:
        # Group consecutive frames
        frame_groups = []
        current_group = [background_frames[0]]
        
        for i in range(1, len(background_frames)):
            if background_frames[i] == background_frames[i-1] + 1:
                current_group.append(background_frames[i])
            else:
                if len(current_group) >= 10:  # Minimum length
                    frame_groups.append(current_group)
                current_group = [background_frames[i]]
        
        if len(current_group) >= 10:  # Add last group if long enough
            frame_groups.append(current_group)
        
        # Convert frame groups to sample segments
        for group in frame_groups:
            start = group[0] * hop_length
            end = (group[-1] + 1) * hop_length
            
            # Ensure minimum segment length
            if end - start >= sr * 0.5:  # At least 0.5 seconds
                background_segments.append((start, end))
    
    return background_segments

# Extract mel spectrograms for CNN
def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram suitable for CNN processing"""
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

# Adaptive processing pipeline - handles different instrument types appropriately
def process_audio_for_classification(audio, sr, instrument_hint=None):
    """Process audio differently based on instrument type hint"""
    
    # For sing and other percussive instruments
    if instrument_hint == 'sing':
        # Use onset detection approach
        segments, segment_times = detect_percussion_events(audio, sr)
        
        # If no clear onsets detected, fall back to standard processing
        if len(segments) == 0:
            return process_audio_standard(audio, sr)
        
        # Process each segment to extract features
        segment_features = []
        for segment in segments[:Config.MAX_SEGMENTS]:  # Limit to max segments
            mel_spec = extract_mel_spectrogram(segment, sr)
            segment_features.append(mel_spec)
        
        # Pad if needed
        while len(segment_features) < Config.MAX_SEGMENTS:
            if segment_features:
                zero_segment = np.zeros_like(segment_features[0])
            else:
                # Fall back to standard processing if no segments created
                return process_audio_standard(audio, sr)
            segment_features.append(zero_segment)
        
        return np.stack(segment_features[:Config.MAX_SEGMENTS]), segment_times
    
    # For background sounds (non-musical)
    elif instrument_hint == 'background':
        # Identify background/silence segments
        background_segments = identify_background_segments(audio, sr)
        
        # If no clear segments, use standard processing
        if len(background_segments) == 0:
            return process_audio_standard(audio, sr)
        
        # Extract background segments
        all_segments = []
        segment_times = []
        
        for start, end in background_segments[:Config.MAX_SEGMENTS]:
            segment = audio[start:end]
            
            # Ensure segment is long enough
            if len(segment) >= sr * 0.5:  # At least 0.5 seconds
                all_segments.append(segment)
                segment_times.append((start/sr, end/sr))
        
        # Process segments
        segment_features = []
        for segment in all_segments[:Config.MAX_SEGMENTS]:
            # Ensure fixed length
            if len(segment) < sr * Config.SEGMENT_DURATION:
                segment = np.pad(segment, (0, sr * Config.SEGMENT_DURATION - len(segment)))
            else:
                segment = segment[:sr * Config.SEGMENT_DURATION]
            
            mel_spec = extract_mel_spectrogram(segment, sr)
            segment_features.append(mel_spec)
        
        # Pad if needed
        while len(segment_features) < Config.MAX_SEGMENTS:
            if segment_features:
                zero_segment = np.zeros_like(segment_features[0])
                segment_features.append(zero_segment)
            else:
                return process_audio_standard(audio, sr)
        
        return np.stack(segment_features[:Config.MAX_SEGMENTS]), segment_times
    
    # Default processing for other instruments
    else:
        return process_audio_standard(audio, sr)

# Standard processing with fixed overlapping windows
def process_audio_standard(audio, sr):
    """Process audio using overlapping windows and create 2D spectrograms"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, Config.MAX_SEGMENTS)
    
    # Extract spectrograms for each segment
    segment_specs = []
    segment_times = []
    
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
        segment_times.append((start/sr, end/sr))
    
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
        segment_times.append(None)  # No time for padded segments
    
    # Stack segments into a single array
    # Result will have shape (MAX_SEGMENTS, n_mels, time_steps)
    return np.stack(segment_specs[:Config.MAX_SEGMENTS]), segment_times

# Function to load and process data for CNN with improved handling
def load_data_for_cnn():
    print("Loading and processing data for CNN model...")
    features_list = []  # Will hold spectrograms for each audio file
    labels = []
    file_paths = []  # Store file paths for reference
    
    # Identify parent folders (assuming each instrument has multiple session folders)
    parent_dirs = [d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    print(f"Found {len(parent_dirs)} instrument categories: {parent_dirs}")
    
    # Create a mapping of folder names to class labels
    instrument_folders = []
    for parent in parent_dirs:
        if parent.startswith('khean-') or parent.startswith('khean_') or parent == 'khean' or parent == 'khaen':
            instrument_class = 'khean'
        elif parent.startswith('khong-vong-') or parent.startswith('khong_vong_') or parent == 'khong_wong' or parent == 'khong_vong':
            instrument_class = 'khong_vong'
        elif parent.startswith('pin-') or parent.startswith('pin_') or parent == 'pin':
            instrument_class = 'pin'
        elif parent.startswith('ra-nad-') or parent.startswith('ra_nad_') or parent == 'ranad':
            instrument_class = 'ranad'
        elif parent.startswith('saw-') or parent.startswith('saw_') or parent == 'so_u' or parent == 'saw':
            instrument_class = 'saw'
        elif parent.startswith('sing-') or parent.startswith('sing_') or parent == 'sing':
            instrument_class = 'sing'
        elif parent.startswith('background-') or parent.startswith('background_') or 'noise' in parent.lower():
            instrument_class = 'background'
        else:
            instrument_class = parent.split('-')[0].lower()
        
        instrument_folders.append((parent, instrument_class))
    
    # Get unique instrument classes
    instrument_classes = sorted(list(set([cls for _, cls in instrument_folders])))
    print(f"Identified {len(instrument_classes)} unique instrument classes: {instrument_classes}")
    
    # Process each folder
    for folder, instrument_class in tqdm(instrument_folders, desc="Processing folders"):
        folder_path = os.path.join(Config.DATA_PATH, folder)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"  Found {len(audio_files)} files in {folder} ({instrument_class})")
        
        for audio_file in tqdm(audio_files, desc=f"Processing {folder}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio file (without duration limit to handle longer files)
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files (less than 1 second)
                if len(audio) < sr:
                    print(f"  Skipping {file_path} - too short ({len(audio)/sr:.2f}s)")
                    continue
                
                # Skip empty or corrupted files
                if np.max(np.abs(audio)) < 0.01:
                    print(f"  Skipping {file_path} - likely silence/corrupted")
                    continue
                
                # Determine instrument hint for adaptive processing
                if instrument_class == 'sing':
                    instrument_hint = 'sing'
                elif instrument_class == 'background':
                    instrument_hint = 'background'
                else:
                    instrument_hint = None
                
                # Process audio with appropriate method
                specs, segment_times = process_audio_for_classification(audio, sr, instrument_hint)
                
                # Add to dataset
                features_list.append(specs)
                labels.append(instrument_class)
                file_paths.append(file_path)
                
                # Apply data augmentation
                augmented_samples = enhanced_augment_audio(audio, sr, instrument_class)
                
                # Process augmented samples
                for aug_audio in augmented_samples:
                    # Process with same instrument hint
                    aug_specs, _ = process_audio_for_classification(aug_audio, sr, instrument_hint)
                    features_list.append(aug_specs)
                    labels.append(instrument_class)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    labels = np.array(labels)
    
    print(f"Finished processing data:")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Each sample has {Config.MAX_SEGMENTS} segments")
    if features_list:
        print(f"  Spectrogram shape: {features_list[0].shape}")
    print(f"  Class distribution:")
    for cls in instrument_classes:
        count = np.sum(labels == cls)
        print(f"    {cls}: {count} samples ({count / len(labels) * 100:.1f}%)")
    
    return features_list, labels, instrument_classes, file_paths

# Build improved CNN model with attention, optimized for Intel GPU
def build_improved_cnn_with_attention(input_shape, num_classes):
    """Build an improved CNN model with residual connections and attention"""
    # Input for segments - shape: (MAX_SEGMENTS, n_mels, time_steps)
    segments_input = tf.keras.layers.Input(shape=input_shape)
    
    # Build CNN feature extractor with residual connections
    def create_segment_encoder():
        inputs = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2]))
        
        # Add channel dimension for 2D convolutions
        x = tf.keras.layers.Reshape((input_shape[1], input_shape[2], 1))(inputs)
        
        # First block with residual connection
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        drop1 = tf.keras.layers.Dropout(0.2)(pool1)
        
        # Second block with residual connection
        conv2a = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(drop1)
        conv2a = tf.keras.layers.BatchNormalization()(conv2a)
        
        # Parallel path with different kernel size
        conv2b = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(drop1)
        conv2b = tf.keras.layers.BatchNormalization()(conv2b)
        
        # Combine parallel paths
        conv2_combined = tf.keras.layers.add([conv2a, conv2b])
        conv2_combined = tf.keras.layers.Activation('relu')(conv2_combined)
        
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2_combined)
        drop2 = tf.keras.layers.Dropout(0.3)(pool2)
        
        # Add frequency attention mechanism - focus on important frequency bands
        freq_attention = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(drop2)
        drop2_att = tf.keras.layers.multiply([drop2, freq_attention])
        
        # Third block with residual connection
        conv3a = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(drop2_att)
        conv3a = tf.keras.layers.BatchNormalization()(conv3a)
        
        # Skip connection from previous layer (with projection to match dimensions)
        skip_proj = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(drop2_att)
        skip_proj = tf.keras.layers.BatchNormalization()(skip_proj)
        
        conv3_combined = tf.keras.layers.add([conv3a, skip_proj])
        conv3_combined = tf.keras.layers.Activation('relu')(conv3_combined)
        
        pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3_combined)
        drop3 = tf.keras.layers.Dropout(0.4)(pool3)
        
        # Fourth block - deeper features
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(drop3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(pool4)
        
        # Global pooling with both average and max pooling combined
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(drop4)
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(drop4)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        
        return tf.keras.Model(inputs=inputs, outputs=concat)
import os
os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import intel_extension_for_tensorflow as itex
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import random
import json
import seaborn as sns 
import scipy.signal
import onnx
import tf2onnx 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Configure Intel GPU
try:
    # Optimize for Intel GPUs
    itex.set_backend('gpu')
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPUs")
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Set memory growth for {device}")
            except Exception as e:
                print(f"Error setting memory growth: {e}")
        
        # Print GPU device info
        print("GPU device information:")
        for device in physical_devices:
            details = tf.config.experimental.get_device_details(device)
            print(f"  {device}: {details}")
    else:
        print("No GPUs found")
except Exception as e:
    print(f"Error configuring Intel GPU: {e}")

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
    BATCH_SIZE = 16
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
    USE_FOCAL_LOSS = False
    FOCAL_LOSS_GAMMA = 2.0
    
    # Model ensemble
    USE_ENSEMBLE = True
    NUM_ENSEMBLE_MODELS = 3

# Print configuration
print("Lao Instrument Classification - Improved CNN Model Training")
print(f"TensorFlow version: {tf.__version__}")
print(f"Librosa version: {librosa.__version__}")
print("-" * 50)

# Print configuration details
print("Configuration:")
for attr in dir(Config):
    if not attr.startswith("__"):
        print(f"  {attr}: {getattr(Config, attr)}")
print("-" * 50)

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
                file_paths.append(file_path)  # Append file path for original sample
                
                # Apply data augmentation
                augmented_samples = enhanced_augment_audio(audio, sr, instrument_class)
                
                # Process augmented samples
                for aug_audio in augmented_samples:
                    # Process with same instrument hint
                    aug_specs, _ = process_audio_for_classification(aug_audio, sr, instrument_hint)
                    features_list.append(aug_specs)
                    labels.append(instrument_class)
                    file_paths.append(file_path)  # Append same file path for augmented sample
                
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

# Build improved CNN model with attention and residual connections
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
    
    # Create the segment encoder
    segment_encoder = create_segment_encoder()
    
    # Apply the encoder to each segment using TimeDistributed
    encoded_segments = tf.keras.layers.TimeDistributed(segment_encoder)(segments_input)
    
    # Multi-head attention mechanism for segments
    # This helps the model focus on the most relevant segments
    if Config.MAX_SEGMENTS > 1:
        # Self-attention on the sequence of segment features
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=256, dropout=0.1
        )(encoded_segments, encoded_segments)
        
        # Add skip connection
        attention_output = tf.keras.layers.add([encoded_segments, attention_layer])
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        
        # Add dense layer after attention
        projected = tf.keras.layers.Dense(512, activation='relu')(attention_output)
        projected = tf.keras.layers.Dropout(0.2)(projected)
        
        
        
        # Another skip connection
        projected_output = tf.keras.layers.add([attention_output, projected])
        projected_output = tf.keras.layers.LayerNormalization()(projected_output)
        
        # Global average pooling across segments
        context = tf.keras.layers.GlobalAveragePooling1D()(projected_output)
    else:
        # If only one segment, skip attention mechanism
        context = tf.keras.layers.Flatten()(encoded_segments)
    
    # Final classification layers
    x = tf.keras.layers.Dense(256, activation='relu')(context)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=segments_input, outputs=output)
    
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for multi-class classification with imbalance"""
    gamma = float(gamma)
    alpha = float(alpha)
    
    def focal_loss_fixed(y_true, y_pred):
        """Focal loss function for multi-class classification"""
        # Clip prediction values to avoid log(0) error
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Handle both sparse and one-hot labels using tf.cond and shape checking
        # Create a boolean tensor that's true if y_true has one less dimension than y_pred
        is_sparse = tf.equal(tf.rank(y_true), tf.rank(y_pred) - 1)
        
        # Convert to one-hot if needed, using tf.cond for conditional execution
        y_true_processed = tf.cond(
            is_sparse,
            lambda: tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1]),
            lambda: y_true
        )
        
        # Calculate focal loss
        cross_entropy = -y_true_processed * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.reduce_sum(y_true_processed * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Calculate alpha factor
        alpha_factor = y_true_processed * alpha + (1 - y_true_processed) * (1 - alpha)
        
        # Combine weights with cross entropy
        weighted_ce = focal_weight * tf.reduce_sum(alpha_factor * cross_entropy, axis=-1)
        
        return weighted_ce
    
    return focal_loss_fixed

# Mixup data augmentation function
def apply_mixup(x, y, alpha=0.2):
    """Apply mixup augmentation to batch data"""
    batch_size = tf.shape(x)[0]
    
    # Sample mixup coefficient from beta distribution
    mix_coef = np.random.beta(alpha, alpha, batch_size)
    mix_coef = np.maximum(mix_coef, 1 - mix_coef)  # Ensure we don't completely replace samples
    mix_coef = np.reshape(mix_coef, (batch_size, 1, 1, 1))  # Shape for broadcasting
    
    # Generate random indices for mixing
    indices = np.random.permutation(batch_size)
    
    # Create mixed inputs
    mixed_x = x * mix_coef + x[indices] * (1 - mix_coef)
    
    # Create mixed targets (one-hot form)
    if len(y.shape) == 1:  # If sparse labels, convert to one-hot
        num_classes = len(np.unique(y))
        y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    else:
        y_onehot = y
    
    # Reshape mix_coef for targets
    target_mix_coef = np.reshape(mix_coef, (batch_size, 1))
    mixed_y = y_onehot * target_mix_coef + y_onehot[indices] * (1 - target_mix_coef)
    
    return mixed_x, mixed_y

# Function to implement learning rate scheduling
def get_lr_scheduler():
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.5
        elif epoch < 50:
            return lr * 0.2
        else:
            return lr * 0.1
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Export model to different formats (TFLite and ONNX)
def export_model(model, filepath_base, class_labels=None):
    """Export model to TFLite and ONNX formats"""
    print("Exporting model to different formats...")
    
    # Save TensorFlow model
    model.save(f"{filepath_base}.h5")
    print(f"TensorFlow model saved at: {filepath_base}.h5")
    
    # Export to ONNX
    try:
        onnx_path = f"{filepath_base}.onnx"
        input_signature = [tf.TensorSpec([1] + list(model.input.shape[1:]), tf.float32)]
        
        # Convert model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=onnx_path
        )
        
        print(f"ONNX model saved at: {onnx_path}")
        
        # Save model metadata for easier loading
        model_meta = {
            "input_shape": list(model.input.shape[1:]),  # Remove batch dimension
            "output_shape": list(model.output.shape[1:]),
            "classes": class_labels.tolist() if hasattr(class_labels, 'tolist') else 
                      (list(class_labels) if class_labels is not None else [])
        }
        
        with open(f"{filepath_base}_meta.json", "w") as f:
            json.dump(model_meta, f)
        
        print(f"Model metadata saved at: {filepath_base}_meta.json")
    except Exception as e:
        print(f"Error exporting to ONNX: {str(e)}")
    
    # Export to TFLite
    try:
        # Standard model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Write model to file
        with open(f"{filepath_base}.tflite", "wb") as f:
            f.write(tflite_model)
        
        # Float16 model (better for mobile)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_float16_model = converter.convert()
        
        with open(f"{filepath_base}_float16.tflite", "wb") as f:
            f.write(tflite_float16_model)
        
        print(f"TFLite models saved at: {filepath_base}.tflite and {filepath_base}_float16.tflite")
    except Exception as e:
        print(f"Error exporting to TFLite: {str(e)}")
    
    # Save the class labels for Flutter integration
    with open(f"{filepath_base}_labels.txt", "w") as f:
        for label in class_labels:
            f.write(f"{label}\n")
    
    print(f"Class labels saved at: {filepath_base}_labels.txt")

# Function to train multiple models for ensembling
def train_model_ensemble(X, y, classes, file_paths, num_models=3):
    """Train multiple models with different initializations"""
    models = []
    histories = []
    
    for i in range(num_models):
        print(f"\n=== Training Ensemble Model {i+1}/{num_models} ===")
        
        # Split data with different random seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42+i*10, stratify=y
        )
        
        # Convert data to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        # Build model with modified architecture
        input_shape = X[0].shape
        model = build_improved_cnn_with_attention(input_shape, len(classes))
        
        # Compile with appropriate loss
        if Config.USE_FOCAL_LOSS:
            model.compile(
                optimizer=itex.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss=focal_loss(gamma=Config.FOCAL_LOSS_GAMMA),
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=itex.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Calculate class weights
        if Config.USE_CLASS_WEIGHTS:
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = {i: total_samples / (len(classes) * count) for i, count in enumerate(class_counts)}
            print("Class weights:", class_weights)
        else:
            class_weights = None
        
        # Set up callbacks with model-specific names
        model_save_path = f"{Config.MODEL_SAVE_PATH}_ensemble_{i+1}"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=Config.EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                monitor='val_loss',
                min_lr=0.00001
            ),
            get_lr_scheduler(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_save_path}_best.h5",
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train, len(classes))
        y_test_onehot = tf.keras.utils.to_categorical(y_test, len(classes))
        
        # Train the model
        history = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Model {i+1} test accuracy: {test_acc:.4f}")
        
        # Plot and save confusion matrix for this model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Model {i+1}')
        plt.savefig(f'confusion_matrix_model_{i+1}.png')
        plt.close()
        
        # Add to ensemble
        models.append(model)
        histories.append(history.history)
        
        # Save the model
        export_model(model, model_save_path, class_labels=classes)
    
    return models, histories

# Function to make ensemble predictions
def ensemble_predict(models, X):
    """Make predictions using ensemble of models"""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions from all models
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Function to make predictions and analyze results
def predict_and_analyze(model, X_test, y_test, classes, file_paths=None, models=None):
    """Make predictions and analyze model performance in detail"""
    print("\nDetailed Model Analysis:")
    
    # Make predictions (use ensemble if available)
    if models is not None and len(models) > 1:
        print("Using ensemble prediction with", len(models), "models")
        y_pred = ensemble_predict(models, X_test)
    else:
        y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Basic metrics
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrices
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_final.png')
    plt.close()
    
    # Normalized confusion matrix
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('normalized_confusion_matrix.png')
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        y_test, y_pred_classes, 
        target_names=classes,
        output_dict=True
    )
    
    # Print the report
    print(classification_report(
        y_test, y_pred_classes, 
        target_names=classes
    ))
    
    # Detailed analysis for each class
    print("\nDetailed Analysis by Class:")
    for cls in classes:
        cls_idx = list(classes).index(cls)
        
        # Get metrics from report
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1 = report[cls]['f1-score']
        support = report[cls]['support']
        
        print(f"Class: {cls}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1-score: {f1:.4f}")
        print(f"  - Support: {support}")
        
        # Get correctly and incorrectly classified examples
        mask_true = (y_test == cls_idx)
        mask_correct = (y_test == cls_idx) & (y_pred_classes == cls_idx)
        mask_incorrect = (y_test == cls_idx) & (y_pred_classes != cls_idx)
        
        num_correct = np.sum(mask_correct)
        num_incorrect = np.sum(mask_incorrect)
        
        print(f"  - Correctly classified: {num_correct}/{support} ({num_correct/support*100:.1f}%)")
        print(f"  - Incorrectly classified: {num_incorrect}/{support} ({num_incorrect/support*100:.1f}%)")
        
        # If we have file paths, identify problem files
        if file_paths is not None and mask_incorrect.any():
            incorrect_indices = np.where(mask_incorrect)[0]
            print(f"  - Example misclassified files:")
            
            for i, idx in enumerate(incorrect_indices[:5]):  # Show up to 5 examples
                predicted_class = classes[y_pred_classes[idx]]
                file_path = file_paths[idx] if idx < len(file_paths) else "Unknown"
                print(f"    {i+1}. {os.path.basename(file_path)} - Predicted as: {predicted_class}")
        
        print()
    
    return report, cm, y_pred

# Main function
def main():
    # At the top of your main() function, add:
    Config.USE_FOCAL_LOSS = False
    # Create necessary directories
    create_directory_if_not_exists("models")
    
    # Load and process data
    features_list, labels, classes, file_paths = load_data_for_cnn()
    
    # Convert string labels to numerical indices
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in labels])
    
    # Convert list of features to numpy array
    X = np.array(features_list)
    
    print(f"\nData ready for training:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Classes: {classes}")
    
    # Split into training and testing sets (stratified to maintain class balance)
    X_train, X_test, y_train, y_test, file_paths_train, file_paths_test = train_test_split(
        X, y_encoded, file_paths, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert data to float32 for better compatibility
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # If using ensemble
    if Config.USE_ENSEMBLE:
        print("\nTraining ensemble of models...")
        models, histories = train_model_ensemble(
            X, y_encoded, classes, file_paths, 
            num_models=Config.NUM_ENSEMBLE_MODELS
        )
        
        # Create and save ensemble predictions
        ensemble_predictions = ensemble_predict(models, X_test)
        ensemble_classes = np.argmax(ensemble_predictions, axis=1)
        
        # Evaluate ensemble
        ensemble_accuracy = np.mean(ensemble_classes == y_test)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Detailed analysis
        ensemble_report, ensemble_cm, ensemble_preds = predict_and_analyze(
            models[0], X_test, y_test, classes, file_paths_test, models=models
        )
        
        # Save ensemble model metadata
        ensemble_meta = {
            "num_models": len(models),
            "model_paths": [f"{Config.MODEL_SAVE_PATH}_ensemble_{i+1}.h5" for i in range(len(models))],
            "accuracy": float(ensemble_accuracy),
            "classes": list(classes)
        }
        
        with open(f"{Config.MODEL_SAVE_PATH}_ensemble_meta.json", "w") as f:
            json.dump(ensemble_meta, f)
        
        print(f"Ensemble metadata saved at: {Config.MODEL_SAVE_PATH}_ensemble_meta.json")
        
    else:
        print("\nBuilding CNN model with attention...")
        # Create and compile the model
        model = build_improved_cnn_with_attention(X_train[0].shape, len(classes))
        
        # Use focal loss if configured
        if Config.USE_FOCAL_LOSS:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss=focal_loss(gamma=Config.FOCAL_LOSS_GAMMA),
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Display model summary
        model.summary()
        
        # Calculate class weights for imbalanced data
        if Config.USE_CLASS_WEIGHTS:
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = {i: total_samples / (len(classes) * count) for i, count in enumerate(class_counts)}
            print("Class weights:", class_weights)
        else:
            class_weights = None
        
        print("\nTraining model...")
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=Config.EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                monitor='val_loss',
                min_lr=0.00001
            ),
            get_lr_scheduler(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{Config.MODEL_SAVE_PATH}_best.h5",
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        print("\nEvaluating model...")
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot learning rate
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig('training_history_cnn_improved.png')
        plt.close()
        
        # Detailed analysis
        report, cm, preds = predict_and_analyze(model, X_test, y_test, classes, file_paths_test)
        
        print("\nSaving model...")
        # Save model in various formats
        export_model(model, Config.MODEL_SAVE_PATH, class_labels=classes)
    
    print("\nTesting model on sample files...")
    # Test with a few samples
    if Config.USE_ENSEMBLE:
        test_model_func = lambda x: ensemble_predict(models, x)
        selected_model = models[0]  # Just for file structure inference
    else:
        test_model_func = model.predict
        selected_model = model
    
    # Randomly select test samples from each class
    test_samples = []
    for cls_idx, cls in enumerate(classes):
        # Find samples of this class in test set
        cls_indices = np.where(y_test == cls_idx)[0]
        
        if len(cls_indices) > 0:
            # Select up to 2 samples from each class
            selected_indices = np.random.choice(cls_indices, min(2, len(cls_indices)), replace=False)
            for idx in selected_indices:
                if idx < len(file_paths_test):
                    test_samples.append((file_paths_test[idx], cls))
    
    # Function to predict instrument for a given audio file
    def predict_instrument(audio_path, true_label=None):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Determine instrument hint for processing
            instrument_hint = None
            if "sing" in audio_path.lower():
                instrument_hint = "sing"
            elif "background" in audio_path.lower() or "noise" in audio_path.lower():
                instrument_hint = "background"
            
            # Process audio for CNN
            specs, segment_times = process_audio_for_classification(audio, sr, instrument_hint)
            specs = np.expand_dims(specs, axis=0)  # Add batch dimension
            
            # Make prediction with appropriate model
            prediction = test_model_func(specs)
            if isinstance(prediction, list):
                prediction = prediction[0]  # For ensemble output
                
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get instrument name
            instrument_name = classes[predicted_class]
            
            result = {
                'instrument': instrument_name,
                'confidence': float(confidence),
                'all_probabilities': {
                    classes[i]: float(prob) for i, prob in enumerate(prediction)
                },
                'true_label': true_label
            }
            
            # Calculate entropy as uncertainty measure
            probs = np.array(list(result['all_probabilities'].values()))
            entropy = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(len(probs))
            result['entropy'] = float(entropy)
            
            return result
        except Exception as e:
            print(f"Error predicting {audio_path}: {str(e)}")
            return None
    
    # Test the model
    for test_file, true_label in test_samples:
        result = predict_instrument(test_file, true_label)
        if result:
            correct = result['instrument'] == result['true_label']
            print(f"\nFile: {os.path.basename(test_file)}")
            print(f"True: {result['true_label']}")
            print(f"Predicted: {result['instrument']} with {result['confidence']:.2%} confidence")
            print(f"Uncertainty (entropy): {result['entropy']:.4f}")
            print(f"Correct: {'✓' if correct else '✗'}")
            print("All probabilities:")
            
            # Sort probabilities by value
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for instr, prob in sorted_probs:
                print(f"  {instr}: {prob:.2%}")
    
    print("\nPipeline completed successfully!")
    print("Next steps: Deploy the model in your application")

if __name__ == "__main__":
    main()
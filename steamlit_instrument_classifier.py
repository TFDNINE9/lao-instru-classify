import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json

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
else:
    print("No GPUs found, using CPU")

# Configuration optimized for your dataset and Streamlit deployment
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 3.0     # Optimized based on your ~6.7s average duration
    OVERLAP = 0.3              # 30% overlap for better coverage
    MAX_SEGMENTS = 3           # Sufficient for your ~6.7s files
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000
    
    # Preprocessing - SIMPLIFIED TO FIX SHAPE ISSUES
    APPLY_NOISE_REDUCTION = True
    USE_SPECTRAL_CONTRAST = False  # Disabled to fix shape issues
    USE_SPECTRAL_BANDWIDTH = False  # Disabled to fix shape issues
    
    # Model parameters
    BATCH_SIZE = 16
    EPOCHS = 150
    LEARNING_RATE = 0.0003     # Lower learning rate for stability
    EARLY_STOPPING_PATIENCE = 20
    
    # Data augmentation
    NUM_AUGMENTATIONS = {      # Class-specific augmentation counts
        'background': 3,       # More augmentation for underrepresented class
        'default': 1           # Default for other classes
    }
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/lao_instruments_model_optimized"
    
    # Training options
    USE_CLASS_WEIGHTS = True
    USE_CROSS_VALIDATION = True
    CV_FOLDS = 5
    
    # Export options
    EXPORT_ONNX = True         # Export to ONNX format for web deployment

print("\nLao Instrument Classification - Optimized for Streamlit Web Deployment")
print(f"TensorFlow version: {tf.__version__}")
print("-" * 50)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Adaptive noise reduction based on signal characteristics
def reduce_noise(audio, sr):
    """Apply adaptive noise reduction based on signal analysis"""
    if not Config.APPLY_NOISE_REDUCTION:
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
            noise_spec = np.abs(librosa.stft(noise_segment, n_fft=Config.N_FFT))
            
            # Compute signal spectrum
            signal_spec = librosa.stft(audio, n_fft=Config.N_FFT)
            signal_mag = np.abs(signal_spec)
            signal_phase = np.angle(signal_spec)
            
            # Spectral subtraction with adaptive threshold
            # More aggressive for background class, more conservative for instruments
            strength = 1.5
            spec_sub = np.maximum(
                signal_mag - strength * noise_spec.mean(axis=1, keepdims=True), 
                0.01 * signal_mag
            )
            
            # Reconstruct signal
            audio_denoised = librosa.istft(spec_sub * np.exp(1j * signal_phase), hop_length=Config.HOP_LENGTH)
            
            # Ensure same length
            if len(audio_denoised) < len(audio):
                audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
            else:
                audio_denoised = audio_denoised[:len(audio)]
                
            return audio_denoised
    
    return audio

# Instrument-specific data augmentation
def augment_for_instrument(audio, sr, instrument_class):
    """Apply targeted augmentations based on instrument characteristics"""
    augmented_samples = []
    
    # Determine number of augmentations
    num_augmentations = Config.NUM_AUGMENTATIONS.get(
        instrument_class, Config.NUM_AUGMENTATIONS['default']
    )
    
    for i in range(num_augmentations):
        aug_audio = audio.copy()
        
        # 1. Time stretching - more subtle than original code
        if instrument_class in ['khean', 'saw']:
            # Wind/string instruments - subtle time stretching
            stretch_factor = np.random.uniform(0.95, 1.05)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        elif instrument_class in ['sing', 'khong_vong', 'ranad']:
            # Percussion instruments - wider time stretching
            stretch_factor = np.random.uniform(0.9, 1.1)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
            
        # 2. Pitch shifting - instrument-specific
        if instrument_class in ['khean', 'saw', 'pin']:
            # Instruments with clear pitch - apply pitch shifting
            pitch_shift = np.random.uniform(-1, 1)  # More subtle pitch shift
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=pitch_shift)
            
        # 3. Background-specific augmentation
        if instrument_class == 'background':
            # Add more variety to background sounds
            noise_factor = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(len(aug_audio))
            aug_audio = aug_audio + noise_factor * noise
            
        # 4. Common to all - volume adjustment
        volume_factor = np.random.uniform(0.8, 1.2)
        aug_audio = aug_audio * volume_factor
        
        # Ensure consistent length
        if len(aug_audio) > len(audio):
            aug_audio = aug_audio[:len(audio)]
        elif len(aug_audio) < len(audio):
            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

# Extract enhanced features for instrument classification - SIMPLIFIED VERSION
def extract_optimized_features(audio, sr):
    """Extract features optimized for Lao instrument classification"""
    # Apply noise reduction
    audio = reduce_noise(audio, sr)
    
    # Extract mel spectrogram
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
    
    # Add channel dimension for CNN input (batch_size, height, width, channels)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
    return mel_spec_db

# Process audio with overlapping segments
def process_audio_segments(audio, sr):
    """Process audio using overlapping segments"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, Config.MAX_SEGMENTS)
    
    # Extract features for each segment
    segment_features = []
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        # Ensure we don't go beyond the audio length
        if end > len(audio):
            break
            
        segment = audio[start:end]
        
        # Extract optimized features
        features = extract_optimized_features(segment, sr)
        segment_features.append(features)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_features) < Config.MAX_SEGMENTS:
        if segment_features:
            zero_segment = np.zeros_like(segment_features[0])
            segment_features.append(zero_segment)
        else:
            # Should not happen with normal audio files
            print("Warning: Failed to extract any segments!")
            # Create dummy feature set
            time_steps = segment_length // Config.HOP_LENGTH + 1
            dummy_feature = np.zeros((Config.N_MELS, time_steps, 1))  # Single channel
            segment_features.append(dummy_feature)
    
    return np.array(segment_features[:Config.MAX_SEGMENTS])

# Load and process the dataset
def load_data_for_classification():
    print("Loading and processing data...")
    features_list = []
    labels = []
    file_paths = []
    
    # Identify instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    print(f"Found {len(instrument_folders)} instrument folders: {instrument_folders}")
    
    # Map folders to standard instrument names
    instrument_mapping = {}
    for folder in instrument_folders:
        folder_lower = folder.lower()
        if 'khean' in folder_lower or 'khaen' in folder_lower:
            instrument_mapping[folder] = 'khean'
        elif 'khong' in folder_lower or 'kong' in folder_lower:
            instrument_mapping[folder] = 'khong_vong'
        elif 'pin' in folder_lower:
            instrument_mapping[folder] = 'pin'
        elif 'nad' in folder_lower or 'ranad' in folder_lower:
            instrument_mapping[folder] = 'ranad'
        elif 'saw' in folder_lower or 'so' in folder_lower:
            instrument_mapping[folder] = 'saw'
        elif 'sing' in folder_lower:
            instrument_mapping[folder] = 'sing'
        elif 'background' in folder_lower or 'noise' in folder_lower:
            instrument_mapping[folder] = 'background'
        else:
            instrument_mapping[folder] = folder_lower
    
    print("Folder to instrument mapping:")
    for folder, instrument in instrument_mapping.items():
        print(f"  {folder} -> {instrument}")
    
    # Track class counts for reporting
    class_counts = {}
    
    # Process each folder
    for folder in tqdm(instrument_folders, desc="Processing folders"):
        instrument = instrument_mapping[folder]
        class_counts[instrument] = class_counts.get(instrument, 0)
        
        folder_path = os.path.join(Config.DATA_PATH, folder)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        print(f"  Found {len(audio_files)} files in {folder} ({instrument})")
        
        for audio_file in tqdm(audio_files, desc=f"Processing {instrument}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:  # Shorter than 0.5 seconds
                    print(f"  Skipping {file_path} - too short ({len(audio)/sr:.2f}s)")
                    continue
                
                # Process audio with segmentation
                specs = process_audio_segments(audio, sr)
                
                # Add to dataset
                features_list.append(specs)
                labels.append(instrument)
                file_paths.append(file_path)
                class_counts[instrument] += 1
                
                # Apply augmentation
                augmented_samples = augment_for_instrument(audio, sr, instrument)
                
                # Process augmented samples
                for aug_audio in augmented_samples:
                    aug_specs = process_audio_segments(aug_audio, sr)
                    features_list.append(aug_specs)
                    labels.append(instrument)
                    file_paths.append(file_path)  # Same file path for augmented sample
                    class_counts[instrument] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels)
    
    print(f"Finished processing data:")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Class distribution:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count} samples ({count / len(labels) * 100:.1f}%)")
    
    return features_array, labels_array, list(set(labels)), file_paths

# Build simplified model for Lao instrument classification
def build_instrument_classifier(input_shape, num_classes):
    """Build model optimized for Lao instrument classification"""
    # Input shape: (MAX_SEGMENTS, n_mels, time_steps, features)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Process each segment with a shared CNN
    def create_segment_processor():
        segment_input = tf.keras.layers.Input(shape=input_shape[1:])
        
        # First convolutional block
        x = tf.keras.layers.Conv2D(32, (5, 5), padding='same')(segment_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Second convolutional block
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third convolutional block
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        return tf.keras.Model(inputs=segment_input, outputs=x)
    
    # Create the segment processor
    segment_processor = create_segment_processor()
    
    # Process each segment
    processed_segments = tf.keras.layers.TimeDistributed(segment_processor)(inputs)
    
    # Self-attention mechanism for multiple segments
    if Config.MAX_SEGMENTS > 1:
        # Add attention to focus on the most informative segments
        attention = tf.keras.layers.Dense(128, activation='tanh')(processed_segments)
        attention = tf.keras.layers.Dense(1, activation=None)(attention)
        attention_weights = tf.keras.layers.Softmax(axis=1)(
            tf.keras.layers.Reshape((Config.MAX_SEGMENTS,))(attention)
        )
        
        # Apply attention weights to get a weighted sum of segment features
        context_vector = tf.keras.layers.Dot(axes=(1, 1))([
            attention_weights, 
            tf.keras.layers.Reshape((Config.MAX_SEGMENTS, -1))(processed_segments)
        ])
        
        x = tf.keras.layers.Flatten()(context_vector)
    else:
        x = tf.keras.layers.Flatten()(processed_segments)
    
    # Final classification layers
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Train and evaluate the model
def train_and_evaluate_model(X, y, classes):
    """Train and evaluate the model"""
    # Convert string labels to numerical indices
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in y])
    
    # Convert to one-hot encoding for training
    y_onehot = tf.keras.utils.to_categorical(y_encoded, len(classes))
    
    # Get input shape
    input_shape = X[0].shape
    print(f"Input shape: {input_shape}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
        X, y_encoded, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Build model
    model = build_instrument_classifier(input_shape, len(classes))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Calculate class weights if enabled
    if Config.USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(classes) * count) for i, count in enumerate(class_counts)}
        print("Class weights:", class_weights)
    else:
        class_weights = None
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=7, 
            monitor='val_loss',
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{Config.MODEL_SAVE_PATH}_best.h5",
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_optimized.png')
    plt.close()
    
    # Generate and plot confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_optimized.png')
    plt.close()
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    return model, history, report, class_to_index

# Export model to ONNX format for web deployment
def export_to_onnx(model, filepath, class_labels, class_indices):
    """Export model to ONNX format for Streamlit web deployment"""
    print("\nExporting model to ONNX format...")
    
    create_directory_if_not_exists(os.path.dirname(filepath))
    
    try:
        # Use tf2onnx for conversion
        import tf2onnx
        
        # Define input signature
        input_signature = [tf.TensorSpec(
            shape=[1] + list(model.input.shape[1:]), 
            dtype=tf.float32, 
            name='input'
        )]
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=f"{filepath}.onnx"
        )
        
        print(f"ONNX model saved at: {filepath}.onnx")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        print("Continuing without ONNX export")
    
    # Save metadata
    model_meta = {
        "input_shape": list(model.input.shape[1:]),
        "output_shape": list(model.output.shape[1:]),
        "classes": class_labels,
        "class_indices": class_indices
    }
    
    with open(f"{filepath}_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)
    
    print(f"Model metadata saved at: {filepath}_meta.json")
    
    # Save preprocessing parameters
    preprocess_params = {
        "sample_rate": Config.SAMPLE_RATE,
        "segment_duration": Config.SEGMENT_DURATION,
        "overlap": Config.OVERLAP,
        "max_segments": Config.MAX_SEGMENTS,
        "n_mels": Config.N_MELS,
        "n_fft": Config.N_FFT,
        "hop_length": Config.HOP_LENGTH,
        "fmax": Config.FMAX,
        "use_noise_reduction": Config.APPLY_NOISE_REDUCTION,
        "use_spectral_contrast": Config.USE_SPECTRAL_CONTRAST,
        "use_spectral_bandwidth": Config.USE_SPECTRAL_BANDWIDTH
    }
    
    with open(f"{filepath}_preprocess.json", "w") as f:
        json.dump(preprocess_params, f, indent=2)
    
    print(f"Preprocessing parameters saved at: {filepath}_preprocess.json")

# Main function
def main():
    # Create necessary directories
    create_directory_if_not_exists("models")
    
    # Load and process data
    X, y, classes, file_paths = load_data_for_classification()
    
    print(f"\nData loaded and processed:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Number of classes: {len(classes)}")
    print(f"  - Classes: {classes}")
    
    # Train and evaluate model
    model, history, report, class_indices = train_and_evaluate_model(X, y, classes)
    
    # Save model in HDF5 format
    create_directory_if_not_exists(os.path.dirname(Config.MODEL_SAVE_PATH))
    model.save(f"{Config.MODEL_SAVE_PATH}.h5")
    print(f"Model saved at: {Config.MODEL_SAVE_PATH}.h5")
    
    # Export to ONNX for web deployment
    if Config.EXPORT_ONNX:
        export_to_onnx(model, Config.MODEL_SAVE_PATH, classes, class_indices)
    
    # Save report as JSON
    with open(f"{Config.MODEL_SAVE_PATH}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved at: {Config.MODEL_SAVE_PATH}_report.json")
    
    print("\nTraining and evaluation completed successfully!")
    print("The optimized model is ready for Streamlit web deployment.")

if __name__ == "__main__":
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import random
import json
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import scipy.signal

# Configure GPU if available
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

# Configuration parameters
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 3      # Shorter segments for more precise classification
    OVERLAP = 0.4             # Less overlap to create more diverse segments
    MAX_SEGMENTS = 4          # Fewer segments to reduce computational load
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000
    
    # Model parameters
    BATCH_SIZE = 16           # Increased for better gradient estimates
    EPOCHS = 100
    LEARNING_RATE = 0.0005    # Reduced for more stable training
    EARLY_STOPPING_PATIENCE = 12
    
    # Data augmentation - SIMPLIFIED
    NUM_AUGMENTATIONS = 1     # Reduced to prevent overfitting
    
    # Paths
    DATA_PATH = "dataset"  # Update to your data folder path
    MODEL_SAVE_PATH = "lao_instruments_model_optimized"
    
    # Training options
    USE_CLASS_WEIGHTS = True
    USE_MIXUP = False         # Disabled as it may confuse instrument timbres
    USE_FOCAL_LOSS = True     # Enabled to focus on hard examples
    FOCAL_LOSS_GAMMA = 2.0
    
    # Cross-validation
    USE_CROSS_VALIDATION = True
    CV_FOLDS = 5
    
    # Preprocessing
    APPLY_NOISE_REDUCTION = True
    USE_DELTA_FEATURES = True  # Include delta features for better temporal info

# Print configuration
print("Lao Instrument Classification - Optimized Model")
print(f"TensorFlow version: {tf.__version__}")
print(f"Librosa version: {librosa.__version__}")
print("-" * 50)

# Print configuration details
print("Configuration:")
for attr in dir(Config):
    if not attr.startswith("__"):
        print(f"  {attr}: {getattr(Config, attr)}")

# Create directory if not exists
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Noise reduction function
def reduce_noise(audio, sr):
    """Apply simple noise reduction"""
    if not Config.APPLY_NOISE_REDUCTION:
        return audio
    
    # Estimate noise profile from quietest part of the signal
    frame_length = 2048
    hop_length = 512
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find the quietest 10% of the signal for noise profile
    percentile_val = np.percentile(rms, 10)
    noise_frames = np.where(rms <= percentile_val)[0]
    
    if len(noise_frames) > 0:
        # Extract noise segment
        noise_start = noise_frames[0] * hop_length
        noise_end = noise_frames[-1] * hop_length + frame_length
        noise_segment = audio[noise_start:min(noise_end, len(audio))]
        
        # Spectral subtraction (simple version)
        S_audio = librosa.stft(audio, n_fft=2048)
        S_noise = librosa.stft(noise_segment, n_fft=2048)
        
        # Compute noise profile
        noise_profile = np.mean(np.abs(S_noise), axis=1, keepdims=True)
        
        # Subtract noise with a threshold to avoid negative values
        S_clean = S_audio - 2 * noise_profile
        S_clean = np.maximum(S_clean, 0.01 * np.abs(S_audio))
        
        # Reconstruct the audio
        audio_clean = librosa.istft(S_clean)
        
        # Match length
        if len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
        else:
            audio_clean = audio_clean[:len(audio)]
        
        return audio_clean
    
    return audio

# Focused data augmentation based on instrument characteristics
def focused_augment_audio(audio, sr, instrument_class=None):
    """Apply targeted augmentations based on instrument type"""
    augmented_samples = []
    
    # Determine number of augmentations
    num_augmentations = Config.NUM_AUGMENTATIONS
    
    for i in range(num_augmentations):
        aug_audio = audio.copy()
        
        # Instrument-specific augmentation
        if instrument_class == 'khean' or instrument_class == 'saw':
            # Wind/string instruments - pitch and subtle time stretching
            if random.random() > 0.5:
                # Subtle pitch shifting (±1 semitone)
                pitch_shift = np.random.uniform(-1, 1)
                aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=pitch_shift)
        
        elif instrument_class == 'sing' or instrument_class == 'khong_vong' or instrument_class == 'ranad':
            # Percussion instruments - timing variations
            if random.random() > 0.5:
                # Subtle time stretching
                stretch_factor = np.random.uniform(0.95, 1.05)
                aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        
        elif instrument_class == 'background':
            # Background noise - add more background variations
            if random.random() > 0.5:
                noise_factor = np.random.uniform(0.01, 0.05)
                noise = np.random.randn(len(aug_audio))
                aug_audio = aug_audio + noise_factor * noise / np.std(noise)
        
        # Common augmentation for all - volume adjustment
        if random.random() > 0.5:
            volume_factor = np.random.uniform(0.8, 1.2)
            aug_audio = aug_audio * volume_factor
            
        # Ensure consistent length
        if len(aug_audio) > len(audio):
            aug_audio = aug_audio[:len(audio)]
        elif len(aug_audio) < len(audio):
            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

# Extract features optimized for instrument classification
def extract_optimized_features(audio, sr):
    """Extract multiple features optimized for instrument classification"""
    # Basic mel spectrogram
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
    
    if Config.USE_DELTA_FEATURES:
        # Add delta features to capture temporal changes
        mel_delta = librosa.feature.delta(mel_spec_db)
        
        # Stack features
        combined_features = np.stack([mel_spec_db, mel_delta], axis=-1)
        return combined_features
    else:
        # Add channel dimension for CNN
        return np.expand_dims(mel_spec_db, axis=-1)

# Process audio with adaptive handling for different instruments
def process_audio_for_classification(audio, sr, instrument_hint=None):
    """Process audio with instrument-specific adaptations"""
    # Apply noise reduction as preprocessing
    audio = reduce_noise(audio, sr)
    
    # Specialized handling for percussive instruments
    if instrument_hint == 'sing' or instrument_hint == 'khong_vong' or instrument_hint == 'ranad':
        # Use onset detection to capture attack transients
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, units='frames', 
            hop_length=Config.HOP_LENGTH,
            backtrack=True
        )
        
        if len(onset_frames) > 0:
            # Use onset information to create segments
            onset_samples = librosa.frames_to_samples(onset_frames, hop_length=Config.HOP_LENGTH)
            segment_specs = []
            
            for onset in onset_samples[:Config.MAX_SEGMENTS]:
                # Create window around onset
                start = max(0, onset - int(0.1 * sr))  # 100ms before
                end = min(len(audio), start + int(Config.SEGMENT_DURATION * sr))
                
                if end - start >= int(0.5 * sr):  # At least 0.5 seconds
                    segment = audio[start:end]
                    
                    # Pad if needed
                    if len(segment) < int(Config.SEGMENT_DURATION * sr):
                        segment = np.pad(segment, (0, int(Config.SEGMENT_DURATION * sr) - len(segment)))
                    
                    # Extract features
                    features = extract_optimized_features(segment, sr)
                    segment_specs.append(features)
            
            # If we found enough segments with onsets
            if len(segment_specs) >= 2:
                # Pad to MAX_SEGMENTS
                while len(segment_specs) < Config.MAX_SEGMENTS:
                    if segment_specs:
                        # Copy an existing segment with slight modification
                        padding_idx = random.randint(0, len(segment_specs) - 1)
                        segment_specs.append(segment_specs[padding_idx] * 0.8)  # Quieter copy
                    else:
                        break
                
                return np.array(segment_specs[:Config.MAX_SEGMENTS])
    
    # Default processing with overlapping windows
    return process_with_overlapping_windows(audio, sr)

def process_with_overlapping_windows(audio, sr):
    """Process audio using overlapping windows"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, Config.MAX_SEGMENTS)
    
    # Extract features for each segment
    segment_specs = []
    
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        # Ensure we don't go beyond the audio length
        if end > len(audio):
            break
            
        segment = audio[start:end]
        
        # Extract features
        features = extract_optimized_features(segment, sr)
        segment_specs.append(features)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_specs) < Config.MAX_SEGMENTS:
        if segment_specs:
            zero_segment = np.zeros_like(segment_specs[0])
            segment_specs.append(zero_segment)
        else:
            # In case we somehow got no segments at all
            time_steps = segment_length // Config.HOP_LENGTH + 1
            feature_dim = 2 if Config.USE_DELTA_FEATURES else 1
            zero_segment = np.zeros((Config.N_MELS, time_steps, feature_dim))
            segment_specs.append(zero_segment)
    
    return np.array(segment_specs[:Config.MAX_SEGMENTS])

# Load and process the dataset
def load_data_for_classification():
    print("Loading and processing data...")
    features_list = []
    labels = []
    file_paths = []  # Store file paths for reference
    
    # Identify parent folders
    parent_dirs = [d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    print(f"Found {len(parent_dirs)} instrument categories: {parent_dirs}")
    
    # Map folder names to class labels
    instrument_folders = []
    for parent in parent_dirs:
        if parent.startswith('khean') or parent.startswith('khaen'):
            instrument_class = 'khean'
        elif parent.startswith('khong-vong') or parent.startswith('khong_vong') or parent.startswith('khong_wong'):
            instrument_class = 'khong_vong'
        elif parent.startswith('pin'):
            instrument_class = 'pin'
        elif parent.startswith('ra-nad') or parent.startswith('ra_nad') or parent.startswith('ranad'):
            instrument_class = 'ranad'
        elif parent.startswith('saw') or parent.startswith('so_u'):
            instrument_class = 'saw'
        elif parent.startswith('sing'):
            instrument_class = 'sing'
        elif parent.startswith('background') or 'noise' in parent.lower():
            instrument_class = 'background'
        else:
            instrument_class = parent.split('-')[0].lower()
        
        instrument_folders.append((parent, instrument_class))
    
    # Get unique instrument classes
    instrument_classes = sorted(list(set([cls for _, cls in instrument_folders])))
    print(f"Identified {len(instrument_classes)} unique instrument classes: {instrument_classes}")
    
    # Process each folder
    class_counts = {}
    for folder, instrument_class in tqdm(instrument_folders, desc="Processing folders"):
        folder_path = os.path.join(Config.DATA_PATH, folder)
        class_counts[instrument_class] = class_counts.get(instrument_class, 0)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"  Found {len(audio_files)} files in {folder} ({instrument_class})")
        
        for audio_file in tqdm(audio_files, desc=f"Processing {folder}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files (less than 1 second)
                if len(audio) < sr:
                    print(f"  Skipping {file_path} - too short ({len(audio)/sr:.2f}s)")
                    continue
                
                # Skip empty or corrupted files
                if np.max(np.abs(audio)) < 0.01:
                    print(f"  Skipping {file_path} - likely silence/corrupted")
                    continue
                
                # Process audio with appropriate method
                specs = process_audio_for_classification(audio, sr, instrument_class)
                
                # Add to dataset
                features_list.append(specs)
                labels.append(instrument_class)
                file_paths.append(file_path)
                class_counts[instrument_class] += 1
                
                # Apply targeted data augmentation
                augmented_samples = focused_augment_audio(audio, sr, instrument_class)
                
                # Process augmented samples
                for aug_audio in augmented_samples:
                    aug_specs = process_audio_for_classification(aug_audio, sr, instrument_class)
                    features_list.append(aug_specs)
                    labels.append(instrument_class)
                    file_paths.append(file_path)  # Same file path for augmented sample
                    class_counts[instrument_class] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    labels = np.array(labels)
    
    print(f"Finished processing data:")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Class distribution:")
    for cls in instrument_classes:
        count = class_counts.get(cls, 0)
        print(f"    {cls}: {count} samples ({count / len(labels) * 100:.1f}%)")
    
    return np.array(features_list), labels, instrument_classes, file_paths

# Build an optimized CNN model for instrument classification
def build_instrument_classifier(input_shape, num_classes):
    """Build a specialized CNN for musical instrument classification"""
    # Input shape: (MAX_SEGMENTS, n_mels, time_steps, features)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Process each segment independently
    def create_segment_processor():
        segment_input = tf.keras.layers.Input(shape=input_shape[1:])
        
        # First convolutional block - capture frequency patterns
        x = tf.keras.layers.Conv2D(32, (5, 5), padding='same')(segment_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Second block - frequency-focused convolutions
        # Vertical filters to capture frequency relationships
        freq_conv = tf.keras.layers.Conv2D(64, (7, 3), padding='same')(x)
        freq_conv = tf.keras.layers.BatchNormalization()(freq_conv)
        freq_conv = tf.keras.layers.LeakyReLU(alpha=0.1)(freq_conv)
        
        # Horizontal filters to capture temporal patterns
        temp_conv = tf.keras.layers.Conv2D(64, (3, 7), padding='same')(x)
        temp_conv = tf.keras.layers.BatchNormalization()(temp_conv)
        temp_conv = tf.keras.layers.LeakyReLU(alpha=0.1)(temp_conv)
        
        # Combine frequency and temporal convolutions
        x = tf.keras.layers.Concatenate()([freq_conv, temp_conv])
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third block - deeper features
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Global pooling - get both average and max features
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
        
        # Combine pooling types
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        return tf.keras.Model(inputs=segment_input, outputs=pooled)
    
    # Create the segment processor
    segment_processor = create_segment_processor()
    
    # Process each segment
    processed_segments = tf.keras.layers.TimeDistributed(segment_processor)(inputs)
    
    # Attention mechanism for multiple segments
    if Config.MAX_SEGMENTS > 1:
        # Self-attention to focus on important segments
        attention = tf.keras.layers.Dense(128, activation='tanh')(processed_segments)
        attention = tf.keras.layers.Dense(1, activation='linear')(attention)
        attention_weights = tf.keras.layers.Softmax(axis=1)(tf.keras.layers.Reshape((Config.MAX_SEGMENTS,))(attention))
        
        # Apply attention weights
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

# Focal loss for better handling of difficult examples
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced classification"""
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert one-hot if needed
        if tf.keras.backend.ndim(y_true) == tf.keras.backend.ndim(y_pred) - 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Calculate alpha factor
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Combine weights with cross entropy
        weighted_loss = focal_weight * tf.reduce_sum(alpha_factor * cross_entropy, axis=-1)
        
        return weighted_loss
    
    return focal_loss_fixed

# Training function
def train_model_with_cross_validation(X, y, classes, file_paths):
    """Train model with cross-validation"""
    # Convert string labels to numerical indices
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in y])
    
    # Get input shape
    input_shape = X[0].shape
    
    # Set up cross-validation
    if Config.USE_CROSS_VALIDATION:
        kfold = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
        fold_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_encoded)):
            print(f"\n--- Training Fold {fold+1}/{Config.CV_FOLDS} ---")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            
            # Convert to float32
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            
            # Create model
            model = build_instrument_classifier(input_shape, len(classes))
            
            # Compile model
            if Config.USE_FOCAL_LOSS:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                    loss=focal_loss(gamma=Config.FOCAL_LOSS_GAMMA),
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Calculate class weights
            if Config.USE_CLASS_WEIGHTS:
                class_counts = np.bincount(y_train)
                total_samples = len(y_train)
                class_weights = {i: total_samples / (len(classes) * count) for i, count in enumerate(class_counts)}
                print(f"Class weights: {class_weights}")
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
                    patience=5, 
                    monitor='val_loss',
                    min_lr=0.00001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{Config.MODEL_SAVE_PATH}_fold_{fold+1}.h5",
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights
            )
            
            # Evaluate model
            _, accuracy = model.evaluate(X_val, y_val)
            fold_scores.append(accuracy)
            models.append(model)
            
            print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
            
            # Plot confusion matrix
            y_pred = np.argmax(model.predict(X_val), axis=1)
            cm = confusion_matrix(y_val, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Fold {fold+1}')
            plt.savefig(f'confusion_matrix_fold_{fold+1}.png')
            plt.close()
        
        print(f"\nCross-validation scores: {fold_scores}")
        print(f"Average accuracy: {np.mean(fold_scores):.4f}")
        
        # Find best model
        best_model_idx = np.argmax(fold_scores)
        best_model = models[best_model_idx]
        print(f"Best model from fold {best_model_idx+1} with accuracy {fold_scores[best_model_idx]:.4f}")
        
        return best_model, fold_scores
    
    else:
        # Standard train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        # Create model
        model = build_instrument_classifier(input_shape, len(classes))
        
        # Compile model
        if Config.USE_FOCAL_LOSS:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss=focal_loss(gamma=Config.FOCAL_LOSS_GAMMA),
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Calculate class weights
        if Config.USE_CLASS_WEIGHTS:
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = {i: total_samples / (len(classes) * count) for i, count in enumerate(class_counts)}
            print(f"Class weights: {class_weights}")
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
                patience=5, 
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
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
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
        plt.savefig('training_history.png')
        plt.close()
        
        # Plot confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        return model, history

# Export model to TFLite
def export_to_tflite(model, filepath, class_labels):
    """Export model to TFLite format for mobile deployment"""
    print("\nExporting model to TFLite format...")
    
    # Standard TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Write model to file
    with open(f"{filepath}.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Standard TFLite model saved at: {filepath}.tflite")
    
    # Float16 model (better for mobile)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_float16_model = converter.convert()
    
    with open(f"{filepath}_float16.tflite", "wb") as f:
        f.write(tflite_float16_model)
    print(f"Float16 TFLite model saved at: {filepath}_float16.tflite")
    
    # Save model input information
    input_shape = model.input.shape.as_list()
    input_info = {
        "max_segments": input_shape[1],
        "segment_feature_dim": np.prod(input_shape[2:]),  # Flatten the remaining dimensions
        "feature_shape": input_shape[2:],
        "total_input_size": np.prod(input_shape[1:])
    }
    
    with open(f"{filepath}_input_info.json", "w") as f:
        json.dump(input_info, f)
    print(f"Input info saved at: {filepath}_input_info.json")
    
    # Save the class labels
    with open(f"{filepath}_labels.txt", "w") as f:
        for label in class_labels:
            f.write(f"{label}\n")
    print(f"Class labels saved at: {filepath}_labels.txt")

# Function to test model on sample files
def test_model_on_samples(model, classes, file_paths, num_samples=5):
    """Test model on a few sample files"""
    print("\nTesting model on sample files...")
    
    # Randomly select test samples from each class
    test_samples = []
    for cls in classes:
        # Find files of this class
        cls_files = [fp for fp, label in zip(file_paths) if label == cls]
        
        if cls_files:
            # Select up to 2 samples from each class
            selected_files = random.sample(cls_files, min(2, len(cls_files)))
            for file_path in selected_files:
                test_samples.append((file_path, cls))
    
    # Function to predict instrument for a given audio file
    def predict_instrument(audio_path, true_label):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Process audio with appropriate method
            specs = process_audio_for_classification(audio, sr, true_label)
            specs = np.expand_dims(specs, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(specs)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get instrument name
            instrument_name = classes[predicted_class]
            
            # Calculate entropy as uncertainty measure
            epsilon = 1e-10
            entropy = -np.sum(prediction * np.log2(prediction + epsilon)) / np.log2(len(prediction))
            
            return {
                'instrument': instrument_name,
                'confidence': float(confidence),
                'entropy': float(entropy),
                'true_label': true_label,
                'all_probabilities': {
                    classes[i]: float(prob) for i, prob in enumerate(prediction)
                }
            }
        except Exception as e:
            print(f"Error predicting {audio_path}: {str(e)}")
            return None
    
    # Test the model
    confusion = {cls: {c: 0 for c in classes} for cls in classes}
    results = []
    
    for test_file, true_label in test_samples:
        result = predict_instrument(test_file, true_label)
        if result:
            correct = result['instrument'] == result['true_label']
            results.append(result)
            
            # Update confusion counts
            confusion[true_label][result['instrument']] += 1
            
            print(f"\nFile: {os.path.basename(test_file)}")
            print(f"True: {result['true_label']}")
            print(f"Predicted: {result['instrument']} with {result['confidence']:.2%} confidence")
            print(f"Uncertainty (entropy): {result['entropy']:.4f}")
            print(f"Correct: {'✓' if correct else '✗'}")
            
            # Print top 3 probabilities
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            print("Top probabilities:")
            for instr, prob in sorted_probs:
                print(f"  {instr}: {prob:.2%}")
    
    # Calculate accuracy on test samples
    if results:
        accuracy = sum(1 for r in results if r['instrument'] == r['true_label']) / len(results)
        print(f"\nTest sample accuracy: {accuracy:.2%}")
    
    return results

# Main function
def main():
    # Create necessary directories
    create_directory_if_not_exists("models")
    
    # Load and process data
    X, y, classes, file_paths = load_data_for_classification()
    
    print(f"\nData loaded and processed:")
    print(f"  - Total samples: {len(X)}")
    if X.size > 0:
        print(f"  - Input shape: {X.shape}")
        print(f"  - Features shape: {X[0].shape}")
    print(f"  - Number of classes: {len(classes)}")
    print(f"  - Classes: {classes}")
    
    # Train model
    model, history = train_model_with_cross_validation(X, y, classes, file_paths)
    
    # Save model
    model.save(f"{Config.MODEL_SAVE_PATH}.h5")
    print(f"Model saved at: {Config.MODEL_SAVE_PATH}.h5")
    
    # Export to TFLite for mobile deployment
    export_to_tflite(model, Config.MODEL_SAVE_PATH, classes)
    
    # Test model on sample files
    test_model_on_samples(model, classes, file_paths)
    
    print("\nTraining and evaluation completed successfully!")
    print("The optimized model should provide better classification accuracy for Lao instruments.")

if __name__ == "__main__":
    main()
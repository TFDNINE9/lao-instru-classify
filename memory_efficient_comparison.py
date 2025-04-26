import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import pandas as pd
import gc  # For garbage collection

# Configure GPU memory growth and limit usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        # You can also limit memory usage to a specific amount
        # tf.config.experimental.set_virtual_device_configuration(
        #     physical_devices[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB limit
        # )
    except Exception as e:
        print(f"GPU configuration error: {e}")

# Configuration for memory-efficient processing
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100  # Your dataset SR
    SEGMENT_DURATION = 2.0  # Reduced from 3.0 to save memory
    OVERLAP = 0.3
    MAX_SEGMENTS = 2  # Reduced from 3 to save memory
    
    # STFT parameters - optimized for memory
    STFT_PARAMS = {
        'n_fft': 1024,  # Reduced from 2048
        'hop_length': 512,  # Increased hop length
        'win_length': 1024,  # Reduced window length
        'window': 'hann'
    }
    
    # Mel spectrogram parameters
    N_MELS = 64  # Reduced from 128
    FMAX = 8000
    
    # Model parameters
    BATCH_SIZE = 16  # Reduced batch size
    EPOCHS = 50  # Reduced epochs for faster testing
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    
    # Data path
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models"
    
    # Cross-validation
    K_FOLDS = 3  # Reduced from 5 for memory efficiency
    
    # Models to compare
    MODELS = ['dnn', 'cnn']  # Removed RNN temporarily to save memory

# Memory-efficient feature extraction
class MemoryEfficientFeatureExtractor:
    def __init__(self, method='mel'):
        self.method = method
    
    def extract_stft_features(self, audio, sr):
        """Extract reduced-size STFT features"""
        # Use smaller FFT parameters
        stft_matrix = librosa.stft(
            audio,
            n_fft=Config.STFT_PARAMS['n_fft'],
            hop_length=Config.STFT_PARAMS['hop_length'],
            win_length=Config.STFT_PARAMS['win_length'],
            window=Config.STFT_PARAMS['window']
        )
        
        # Convert to magnitude
        magnitude = np.abs(stft_matrix)
        
        # Convert to log scale
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Reduce frequency bins by averaging
        # Group every 4 frequency bins together
        bin_size = 4
        reduced_freq = log_magnitude.reshape(log_magnitude.shape[0]//bin_size, bin_size, -1).mean(axis=1)
        
        return reduced_freq
    
    def extract_mel_features(self, audio, sr):
        """Extract mel spectrogram features with reduced dimensions"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.STFT_PARAMS['n_fft'],
            hop_length=Config.STFT_PARAMS['hop_length'],
            fmax=Config.FMAX
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def extract_mfcc_features(self, audio, sr):
        """Extract MFCC features with reduced dimensions"""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20,  # Reduced from 40
            n_fft=Config.STFT_PARAMS['n_fft'],
            hop_length=Config.STFT_PARAMS['hop_length']
        )
        
        # Add only delta features (skip delta-delta to save memory)
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # Stack them together
        combined_mfccs = np.vstack([mfccs, delta_mfccs])
        
        return combined_mfccs
    
    def extract_features(self, audio, sr):
        """Extract features based on method"""
        if self.method == 'stft':
            return self.extract_stft_features(audio, sr)
        elif self.method == 'mel':
            return self.extract_mel_features(audio, sr)
        elif self.method == 'mfcc':
            return self.extract_mfcc_features(audio, sr)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")

# Memory-efficient data preprocessing
def process_audio_segments(audio, sr, feature_extractor):
    """Process audio into segments with features"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Extract features for the entire audio first
    features = feature_extractor.extract_features(audio, sr)
    
    # Calculate frames per segment
    frames_per_segment = int(Config.SEGMENT_DURATION * sr / Config.STFT_PARAMS['hop_length'])
    
    # Extract segments
    segments = []
    for i in range(Config.MAX_SEGMENTS):
        start_frame = i * frames_per_segment // 2  # 50% overlap
        end_frame = start_frame + frames_per_segment
        
        if end_frame <= features.shape[1]:
            segment = features[:, start_frame:end_frame]
        else:
            # Pad if we're at the end
            segment = np.zeros((features.shape[0], frames_per_segment))
            available_frames = features.shape[1] - start_frame
            if available_frames > 0:
                segment[:, :available_frames] = features[:, start_frame:]
        
        segments.append(segment)
    
    return np.array(segments)

# Load data in batches to manage memory
def load_data_in_batches(feature_extractor, batch_size=100):
    """Load data in batches to manage memory"""
    print(f"Loading and processing data in batches with {feature_extractor.method} features...")
    
    # Get all files first
    all_files = []
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
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
    
    # Collect all file paths
    for folder in instrument_folders:
        instrument = instrument_mapping[folder]
        folder_path = os.path.join(Config.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            all_files.append((file_path, instrument))
    
    # Process in batches
    features_list = []
    labels = []
    
    for i in tqdm(range(0, len(all_files), batch_size), desc="Processing batches"):
        batch_files = all_files[i:i+batch_size]
        batch_features = []
        batch_labels = []
        
        for file_path, instrument in batch_files:
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:
                    continue
                
                # Process segments
                segments = process_audio_segments(audio, sr, feature_extractor)
                
                batch_features.append(segments)
                batch_labels.append(instrument)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Add batch to main list
        features_list.extend(batch_features)
        labels.extend(batch_labels)
        
        # Clear memory
        gc.collect()
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y, list(set(labels))

# Simplified model architectures for memory efficiency
def build_simple_dnn_model(input_shape, num_classes):
    """Build a simpler DNN model"""
    # Flatten input for DNN
    flat_input_dim = np.prod(input_shape)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_simple_cnn_model(input_shape, num_classes):
    """Build a simpler CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, 3, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 3, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D()),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Memory-efficient training function
def train_memory_efficient(model_fn, X, y, num_classes, class_names):
    """Train with memory efficiency in mind"""
    # Single split instead of k-fold to save memory
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    model = model_fn(X_train.shape[1:], num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    # Generate confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    # Clear model from memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return test_acc, cm, history

# Main function
def main_memory_efficient():
    """Run memory-efficient experiments"""
    results = {}
    
    # Try different feature extraction methods
    for feature_method in ['mel', 'mfcc']:  # Skip STFT for now due to memory
        print(f"\n{'='*50}")
        print(f"Experiments with {feature_method.upper()} features")
        print(f"{'='*50}")
        
        feature_extractor = MemoryEfficientFeatureExtractor(method=feature_method)
        
        # Load data in batches
        X, y, class_names = load_data_in_batches(feature_extractor)
        
        # Convert labels to integers
        label_to_int = {label: i for i, label in enumerate(class_names)}
        y_encoded = np.array([label_to_int[label] for label in y])
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Classes: {class_names}")
        
        # Try different models
        models = {
            'dnn': build_simple_dnn_model,
            'cnn': build_simple_cnn_model
        }
        
        for model_name, model_fn in models.items():
            print(f"\n{'-'*30}")
            print(f"Training {model_name.upper()} with {feature_method.upper()}")
            print(f"{'-'*30}")
            
            # Train with memory efficiency
            accuracy, cm, history = train_memory_efficient(
                model_fn, X, y_encoded, len(class_names), class_names
            )
            
            # Store results
            results[f"{model_name}_{feature_method}"] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'history': history.history
            }
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{model_name.upper()} with {feature_method.upper()} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'{Config.MODEL_SAVE_PATH}/{model_name}_{feature_method}_confusion_matrix.png')
            plt.close()
            
            # Clear memory after each model
            gc.collect()
    
    # Compare results
    comparison_df = pd.DataFrame([{
        'Model': key.split('_')[0].upper(),
        'Features': key.split('_')[1].upper(),
        'Accuracy': val['accuracy']
    } for key, val in results.items()])
    
    print("\n\nFinal Comparison:")
    print(comparison_df.sort_values('Accuracy', ascending=False))
    
    # Save comparison results
    comparison_df.to_csv(f'{Config.MODEL_SAVE_PATH}/model_comparison.csv', index=False)
    
    return results, comparison_df

if __name__ == "__main__":
    # Create model directory
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Run memory-efficient experiments
    results, comparison_df = main_memory_efficient()
    
    print("\nExperiments completed! Check the 'models' directory for results.")
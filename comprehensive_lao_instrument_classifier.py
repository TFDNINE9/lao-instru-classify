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
from scipy import signal

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error setting memory growth: {e}")

# Configuration for different approaches
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100  # Your dataset SR
    SEGMENT_DURATION = 3.0
    OVERLAP = 0.5
    MAX_SEGMENTS = 3
    
    # STFT parameters (matching your image)
    STFT_PARAMS = {
        'n_fft': 2048,  # Higher for better frequency resolution
        'hop_length': 441,  # About 10ms at 44.1kHz
        'win_length': 1764,  # About 40ms at 44.1kHz
        'window': 'hann'
    }
    
    # Mel spectrogram parameters
    N_MELS = 128
    FMAX = 8000
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    
    # Data path
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models"
    
    # Cross-validation
    K_FOLDS = 5
    
    # Models to train
    MODELS = ['dnn', 'rnn', 'cnn']  # We'll compare all three

# Feature extraction methods
class FeatureExtractor:
    def __init__(self, method='mel'):
        self.method = method
        self.scaler = StandardScaler()
    
    def extract_stft_features(self, audio, sr):
        """Extract STFT features similar to your image"""
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
        
        return log_magnitude
    
    def extract_mel_features(self, audio, sr):
        """Extract mel spectrogram features"""
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
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40,
            n_fft=Config.STFT_PARAMS['n_fft'],
            hop_length=Config.STFT_PARAMS['hop_length']
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack them together
        combined_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
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

# Data preprocessing
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

# Load and preprocess data
def load_data_for_classification(feature_extractor):
    """Load and process the dataset"""
    print(f"Loading and processing data with {feature_extractor.method} features...")
    features_list = []
    labels = []
    file_paths = []
    
    # Identify instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    # Map folders to standard instrument names (using your mapping)
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
    
    # Process each folder
    for folder in tqdm(instrument_folders, desc="Processing folders"):
        instrument = instrument_mapping[folder]
        folder_path = os.path.join(Config.DATA_PATH, folder)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {instrument}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:
                    continue
                
                # Process segments
                segments = process_audio_segments(audio, sr, feature_extractor)
                
                # Add to dataset
                features_list.append(segments)
                labels.append(instrument)
                file_paths.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y, list(set(labels)), file_paths

# Model architectures
def build_dnn_model(input_shape, num_classes):
    """Build a standard DNN model"""
    # Flatten input for DNN
    flat_input_dim = np.prod(input_shape)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_rnn_model(input_shape, num_classes):
    """Build an RNN (LSTM) model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((input_shape[2], input_shape[1]))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(64)),
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

def build_cnn_model(input_shape, num_classes):
    """Build a CNN model (similar to your previous one but simplified)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, 3, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 3, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(128, 3, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D()),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Evaluation functions
def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate a model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get classification report
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    return report, cm

def cross_validate_model(model_fn, X, y, num_classes, class_names):
    """Perform K-fold cross-validation"""
    kfold = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_reports = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{Config.K_FOLDS}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model = model_fn(X_train.shape[1:], num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        report, cm = evaluate_model(model, X_val, y_val, class_names)
        cv_scores.append(report['accuracy'])
        cv_reports.append(report)
        
        print(f"Fold {fold + 1} accuracy: {report['accuracy']:.4f}")
    
    print(f"\nCross-validation results: {cv_scores}")
    print(f"Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return cv_scores, cv_reports

# Main experiment function
def run_experiments():
    """Run comprehensive experiments with different models and features"""
    results = {}
    
    # Try different feature extraction methods
    for feature_method in ['stft', 'mel', 'mfcc']:
        print(f"\n{'='*50}")
        print(f"Experiments with {feature_method.upper()} features")
        print(f"{'='*50}")
        
        feature_extractor = FeatureExtractor(method=feature_method)
        
        # Load data
        X, y, class_names, file_paths = load_data_for_classification(feature_extractor)
        
        # Convert labels to integers
        label_to_int = {label: i for i, label in enumerate(class_names)}
        y_encoded = np.array([label_to_int[label] for label in y])
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Classes: {class_names}")
        
        # Try different models
        models = {
            'dnn': build_dnn_model,
            'rnn': build_rnn_model,
            'cnn': build_cnn_model
        }
        
        for model_name, model_fn in models.items():
            print(f"\n{'-'*30}")
            print(f"Training {model_name.upper()} with {feature_method.upper()}")
            print(f"{'-'*30}")
            
            # Cross-validate
            cv_scores, cv_reports = cross_validate_model(
                model_fn, X, y_encoded, len(class_names), class_names
            )
            
            # Store results
            results[f"{model_name}_{feature_method}"] = {
                'scores': cv_scores,
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'reports': cv_reports
            }
            
            # Train final model on all data for visualization
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            model = model_fn(X_train.shape[1:], len(class_names))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with early stopping
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
                verbose=0
            )
            
            # Evaluate and plot
            report, cm = evaluate_model(model, X_test, y_test, class_names)
            
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
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title(f'{model_name.upper()} with {feature_method.upper()} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title(f'{model_name.upper()} with {feature_method.upper()} - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{Config.MODEL_SAVE_PATH}/{model_name}_{feature_method}_training_history.png')
            plt.close()
    
    # Compare all models
    comparison_df = pd.DataFrame([{
        'Model': key.split('_')[0].upper(),
        'Features': key.split('_')[1].upper(),
        'Mean Accuracy': val['mean_accuracy'],
        'Std Accuracy': val['std_accuracy']
    } for key, val in results.items()])
    
    print("\n\nFinal Comparison:")
    print(comparison_df.sort_values('Mean Accuracy', ascending=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        [f"{row['Model']} + {row['Features']}" for _, row in comparison_df.iterrows()],
        comparison_df['Mean Accuracy'],
        yerr=comparison_df['Std Accuracy'],
        capsize=5
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.savefig(f'{Config.MODEL_SAVE_PATH}/model_comparison.png')
    plt.close()
    
    # Save comparison results
    comparison_df.to_csv(f'{Config.MODEL_SAVE_PATH}/model_comparison.csv', index=False)
    
    # Find the best model
    best_model_idx = comparison_df['Mean Accuracy'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]
    
    print(f"\nBest Model: {best_model['Model']} with {best_model['Features']} features")
    print(f"Mean Accuracy: {best_model['Mean Accuracy']:.4f} ± {best_model['Std Accuracy']:.4f}")
    
    return results, comparison_df

if __name__ == "__main__":
    # Create model directory
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Run experiments
    results, comparison_df = run_experiments()
    
    print("\nExperiments completed! Check the 'models' directory for results.")
import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import tf2onnx
import onnx

# Configuration optimized based on the comparison results
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 3.0
    
    # MFCC parameters - demonstrated to be the best performing feature
    N_MFCC = 40  # Extended from 20 for better frequency resolution
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Feature augmentation
    USE_DELTA = True
    USE_DELTA_DELTA = True
    
    # Data Augmentation
    TIME_STRETCH_RANGE = (0.9, 1.1)
    PITCH_SHIFT_RANGE = (-2, 2)
    NOISE_FACTOR = 0.005
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.0005
    EARLY_STOPPING_PATIENCE = 20
    
    # Regularization
    DROPOUT_RATE = 0.4
    L2_REGULARIZATION = 0.001
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/optimized_mfcc_model"

class OptimizedMFCCExtractor:
    """Optimized MFCC feature extraction based on the comparison results"""
    
    @staticmethod
    def extract_features(audio, sr):
        """Extract MFCC features with deltas"""
        # Ensure audio has the desired length
        desired_length = int(Config.SEGMENT_DURATION * sr)
        if len(audio) > desired_length:
            audio = audio[:desired_length]
        else:
            audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
        
        # Basic MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        
        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        
        features = [mfccs]
        
        # Add delta features
        if Config.USE_DELTA:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta_mfccs = (delta_mfccs - np.mean(delta_mfccs, axis=1, keepdims=True)) / (np.std(delta_mfccs, axis=1, keepdims=True) + 1e-8)
            features.append(delta_mfccs)
        
        # Add delta-delta features
        if Config.USE_DELTA_DELTA:
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            delta2_mfccs = (delta2_mfccs - np.mean(delta2_mfccs, axis=1, keepdims=True)) / (np.std(delta2_mfccs, axis=1, keepdims=True) + 1e-8)
            features.append(delta2_mfccs)
        
        # Stack all features
        combined_features = np.vstack(features)
        
        # Fix the time dimension to a consistent length
        expected_time_frames = int(np.ceil(desired_length / Config.HOP_LENGTH)) + 1
        
        # If we got more frames than expected, truncate
        if combined_features.shape[1] > expected_time_frames:
            combined_features = combined_features[:, :expected_time_frames]
        # If we got fewer frames than expected, pad
        elif combined_features.shape[1] < expected_time_frames:
            pad_width = expected_time_frames - combined_features.shape[1]
            combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
        
        return combined_features

def augment_audio(audio, sr):
    """Apply data augmentation techniques"""
    augmented_samples = []
    
    # Time stretching
    stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
    stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
    augmented_samples.append(stretched)
    
    # Pitch shifting
    pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
    augmented_samples.append(shifted)
    
    # Add noise
    noise = np.random.normal(0, Config.NOISE_FACTOR, len(audio))
    noisy = audio + noise
    augmented_samples.append(noisy)
    
    return augmented_samples

def build_optimized_dnn_model(input_shape, num_classes):
    """Build an optimized DNN model based on comparison results"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First dense block
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Second dense block
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Third dense block
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Fourth dense block
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def process_dataset():
    """Process the dataset with MFCC features and augmentation"""
    print("Processing dataset with optimized MFCC features...")
    
    features_list = []
    labels = []
    
    # Get instrument folders
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
                
                # Process original audio
                features = OptimizedMFCCExtractor.extract_features(audio, sr)
                features_list.append(features)
                labels.append(instrument)
                
                # Apply augmentation
                if instrument != 'background':  # Don't augment background noise
                    augmented_samples = augment_audio(audio, sr)
                    for aug_audio in augmented_samples:
                        aug_features = OptimizedMFCCExtractor.extract_features(aug_audio, sr)
                        features_list.append(aug_features)
                        labels.append(instrument)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Now all features should have the same shape
    X = np.array(features_list)
    
    # Reshape for DNN (flatten features and time)
    X = X.reshape(X.shape[0], -1)
    
    y = np.array(labels)
    
    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, list(unique)

def train_optimized_model(X, y, class_names):
    """Train the optimized model"""
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Compute class weights
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights = dict(enumerate(weights))
        print("\nClass weights:", class_weights)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_optimized_dnn_model(input_shape, len(class_names))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {test_acc:.4f}")
    
    # Generate confusion matrix
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'confusion_matrix.png'))
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'training_history.png'))
    plt.close()
    
    return model, history

def convert_to_onnx(model, model_path):
    """Convert Keras model to ONNX format"""
    # Get model input shape
    input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save ONNX model
    onnx_path = os.path.join(model_path, 'optimized_mfcc_model.onnx')
    onnx.save_model(onnx_model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")
    
    return onnx_path

def main():
    """Main function for optimized MFCC-based classifier"""
    # Create model directory
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Process dataset
    X, y, class_names = process_dataset()
    
    # Train model
    model, history = train_optimized_model(X, y, class_names)
    
    # Save model with metadata
    model_metadata = {
        'class_names': class_names,
        'feature_type': 'mfcc',
        'n_mfcc': Config.N_MFCC,
        'use_delta': Config.USE_DELTA,
        'use_delta_delta': Config.USE_DELTA_DELTA,
        'sample_rate': Config.SAMPLE_RATE,
        'segment_duration': Config.SEGMENT_DURATION,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'input_shape': list(model.input_shape)
    }
    
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Convert to ONNX
    convert_to_onnx(model, Config.MODEL_SAVE_PATH)
    
    print(f"\nTraining complete! Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import random
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Configuration parameters
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 4  # Duration of each segment in seconds
    OVERLAP = 0.5  # 50% overlap between segments
    MAX_SEGMENTS = 5  # Maximum number of segments per audio file
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMAX = 8000
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    
    # Data augmentation
    NUM_AUGMENTATIONS = 2  # Number of augmented samples per original sample
    
    # Paths
    DATA_PATH = "dataset"  # Update this to your data folder path
    MODEL_SAVE_PATH = "lao_instruments_model_dnn_attention"

# Print configuration
print("Lao Instrument Classification - DNN Model with Attention")
print(f"TensorFlow version: {tf.__version__}")
print(f"Librosa version: {librosa.__version__}")
print("-" * 50)

# Utility functions
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Data augmentation functions
def augment_audio(audio, sr):
    """Apply random augmentations to audio sample"""
    augmented_samples = []
    
    for i in range(Config.NUM_AUGMENTATIONS):
        aug_audio = audio.copy()
        
        # 1. Time stretching (0.9-1.1x speed)
        if random.random() > 0.5:
            stretch_factor = np.random.uniform(0.9, 1.1)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        
        # 2. Pitch shifting (up to ±2 semitones)
        if random.random() > 0.5:
            pitch_shift = np.random.uniform(-2, 2)
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=pitch_shift)
        
        # 3. Add background noise (up to 5% of signal amplitude)
        if random.random() > 0.5:
            noise_factor = np.random.uniform(0, 0.05)
            noise = np.random.randn(len(aug_audio))
            aug_audio = aug_audio + noise_factor * noise
        
        # 4. Volume adjustment (0.8-1.2x)
        if random.random() > 0.5:
            volume_factor = np.random.uniform(0.8, 1.2)
            aug_audio = aug_audio * volume_factor
            
        # We don't need to pad/truncate the audio here since we'll be using overlapping windows
        augmented_samples.append(aug_audio)
    
    return augmented_samples

# Feature extraction with overlapping windows
def extract_features_with_overlapping_windows(audio, sr):
    """Extract features from audio using overlapping windows"""
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
        
        # Extract mel spectrogram for this segment
        mel_spec = librosa.feature.melspectrogram(
            y=segment, 
            sr=sr, 
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            fmax=Config.FMAX
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Flatten the mel spectrogram
        flattened_segment = mel_spec_db.flatten()
        segment_features.append(flattened_segment)
    
    # If we have fewer than MAX_SEGMENTS, pad with zeros
    while len(segment_features) < Config.MAX_SEGMENTS:
        # Create a zero-filled array of the same shape as other segments
        if segment_features:
            zero_segment = np.zeros_like(segment_features[0])
        else:
            # In case we somehow got no segments at all
            mel_shape = (Config.N_MELS, segment_length // Config.HOP_LENGTH + 1)
            zero_mel = np.zeros(mel_shape)
            zero_segment = zero_mel.flatten()
        
        segment_features.append(zero_segment)
    
    # Stack segments into a single array
    # Shape will be (MAX_SEGMENTS, feature_dim)
    return np.stack(segment_features[:Config.MAX_SEGMENTS])

# Function to load and process data with overlapping windows
def load_data_with_overlapping_windows():
    print("Loading and processing data with overlapping windows...")
    features_list = []  # Will be a list of arrays, each of shape (MAX_SEGMENTS, feature_dim)
    labels = []
    file_paths = []  # Store file paths for reference
    
    # Identify parent folders (assuming each instrument has multiple session folders)
    parent_dirs = [d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    print(f"Found {len(parent_dirs)} instrument categories: {parent_dirs}")
    
    # Create a mapping of folder names to class labels
    instrument_folders = []
    for parent in parent_dirs:
        if parent.startswith('khean-') or parent.startswith('khean_'):
            instrument_class = 'khean'
        elif parent.startswith('khong-vong-') or parent.startswith('khong_vong_'):
            instrument_class = 'khong_vong'
        elif parent.startswith('pin-') or parent.startswith('pin_'):
            instrument_class = 'pin'
        elif parent.startswith('ra-nad-') or parent.startswith('ra_nad_'):
            instrument_class = 'ranad'
        elif parent.startswith('saw-') or parent.startswith('saw_'):
            instrument_class = 'saw'
        elif parent.startswith('sing-') or parent.startswith('sing_'):
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
                
                # Extract features using overlapping windows
                segments_features = extract_features_with_overlapping_windows(audio, sr)
                features_list.append(segments_features)
                labels.append(instrument_class)
                file_paths.append(file_path)
                
                # Apply data augmentation
                augmented_samples = augment_audio(audio, sr)
                
                # Extract features for augmented samples
                for aug_audio in augmented_samples:
                    aug_segments_features = extract_features_with_overlapping_windows(aug_audio, sr)
                    features_list.append(aug_segments_features)
                    labels.append(instrument_class)
                    # We don't store file paths for augmented samples
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    # features_list is a list of arrays, each of shape (MAX_SEGMENTS, feature_dim)
    labels = np.array(labels)
    
    print(f"Finished processing data:")
    print(f"  Total samples: {len(features_list)}")
    print(f"  Each sample has {Config.MAX_SEGMENTS} segments")
    if features_list:
        print(f"  Each segment has {features_list[0].shape[1]} features")
    print(f"  Class distribution:")
    for cls in instrument_classes:
        count = np.sum(labels == cls)
        print(f"    {cls}: {count} samples ({count / len(labels) * 100:.1f}%)")
    
    return features_list, labels, instrument_classes, file_paths

# Function to build attention-based model for overlapping windows
def build_attention_model(segment_feature_dim, max_segments, num_classes):
    # Input for segments
    segments_input = tf.keras.layers.Input(shape=(max_segments, segment_feature_dim))
    
    # Process each segment independently with the same dense layer (time-distributed)
    segment_encoder = tf.keras.layers.Dense(256, activation='relu')
    encoded_segments = tf.keras.layers.TimeDistributed(segment_encoder)(segments_input)
    
    # Add attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(encoded_segments)
    attention = tf.keras.layers.Reshape((max_segments,))(attention)
    attention_weights = tf.keras.layers.Softmax()(attention)
    
    # Apply attention weights
    context = tf.keras.layers.Dot(axes=1)([attention_weights, encoded_segments])
    
    # Process the combined context
    x = tf.keras.layers.BatchNormalization()(context)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=segments_input, outputs=output)
    
    return model

# Function to implement learning rate scheduling
def get_lr_scheduler():
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.5
        else:
            return lr * 0.1
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Function to convert model to TFLite
def convert_to_tflite(model, filepath_base, max_segments, segment_feature_dim):
    """Convert model to TFLite with explicit input shapes"""
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
    
    # Save the input shape information separately for Flutter
    input_info = {
        "max_segments": int(max_segments),
        "segment_feature_dim": int(segment_feature_dim),
        "total_input_size": int(max_segments * segment_feature_dim)
    }
    
    with open(f"{filepath_base}_input_info.json", "w") as f:
        json.dump(input_info, f)
    
    print(f"Model saved as TFLite at {filepath_base}.tflite")
    print(f"Float16 model saved at {filepath_base}_float16.tflite")
    print(f"Input info saved at {filepath_base}_input_info.json")

# Main function
def main():
    # Create necessary directories
    create_directory_if_not_exists("models")
    
    # Load and process data with overlapping windows
    features_list, labels, classes, file_paths = load_data_with_overlapping_windows()
    
    # Convert string labels to numerical indices
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in labels])
    
    # Get dimensions for model input
    max_segments = Config.MAX_SEGMENTS
    if features_list:
        segment_feature_dim = features_list[0].shape[1]
    else:
        print("Error: No features extracted from audio files!")
        return
    
    # Convert list of features to numpy array
    X = np.array(features_list)
    
    # Split into training and testing sets (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert data to float32 for better compatibility
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print("\nBuilding attention-based model...")
    # Create and compile the model
    model = build_attention_model(segment_feature_dim, max_segments, len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
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
        get_lr_scheduler()
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks
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
    plt.savefig('training_history_attention.png')
    print("Training history saved to training_history_attention.png")
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_attention.png')
    print("Confusion matrix saved to confusion_matrix_attention.png")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_classes, 
        target_names=classes
    ))
    
    print("\nSaving model...")
    # Save standard model
    model.save(f'{Config.MODEL_SAVE_PATH}.h5')
    
    # Save TFLite models
    convert_to_tflite(model, Config.MODEL_SAVE_PATH, max_segments, segment_feature_dim)
    
    # Save the class labels for Flutter integration
    with open('label_encoder.txt', 'w') as f:
        for label in classes:
            f.write(f"{label}\n")
    
    print("Model and label encoder saved successfully!")
    
    print("\nTesting model on sample files...")
    # Test with a few samples
    test_samples = np.random.choice(file_paths, min(5, len(file_paths)), replace=False)
    
    # Function to predict instrument for a given audio file
    def predict_instrument(audio_path):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Extract features with overlapping windows
            segments_features = extract_features_with_overlapping_windows(audio, sr)
            segments_features = np.expand_dims(segments_features, axis=0)
            
            # Make prediction
            prediction = model.predict(segments_features)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get instrument name
            instrument_name = classes[predicted_class]
            
            return {
                'instrument': instrument_name,
                'confidence': float(confidence),
                'all_probabilities': {
                    classes[i]: float(prob) for i, prob in enumerate(prediction)
                }
            }
        except Exception as e:
            print(f"Error predicting {audio_path}: {str(e)}")
            return None
    
    # Test the model
    for test_file in test_samples:
        result = predict_instrument(test_file)
        if result:
            print(f"\nFile: {os.path.basename(test_file)}")
            print(f"Predicted: {result['instrument']} with {result['confidence']:.2%} confidence")
            print("All probabilities:")
            for instr, prob in result['all_probabilities'].items():
                print(f"  {instr}: {prob:.2%}")
    
    print("\nPipeline completed successfully!")
    print("\nNext steps: Update your Flutter application to process audio using overlapping windows")

if __name__ == "__main__":
    main()
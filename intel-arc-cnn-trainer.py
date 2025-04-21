import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import random
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import classes
from tqdm import tqdm
import time

# Intel-specific imports and setup
import tensorflow as tf
try:
    # Try to import Intel extensions
    # import intel_tensorflow as itf
    # import intel_extension_for_tensorflow as itex
    # Configure for Intel GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Intel Arc GPU detected: {physical_devices[0].name}")
    
    # Set Intel optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    print("Intel optimizations enabled")
except ImportError:
    print("Intel TensorFlow extensions not found. Using standard TensorFlow.")

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device name:", tf.test.gpu_device_name())

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
    
    # Intel Arc specific
    USE_MIXED_PRECISION = True  # Enable mixed precision for faster training
    INTEL_OPTIMIZED = True
    
    # Data augmentation
    NUM_AUGMENTATIONS = 2  # Number of augmented samples per original sample
    
    # Paths
    DATA_PATH = "dataset"  # Update this to your data folder path
    MODEL_SAVE_PATH = "models/lao_instruments_model_cnn"

# Enable mixed precision for faster training on Intel GPU
if Config.USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled (float16)")

# Print configuration
print("Lao Instrument Classification - CNN Model Training")
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
            
        augmented_samples.append(aug_audio)
    
    return augmented_samples

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

# Process audio with overlapping windows for CNN
def process_audio_for_cnn(audio, sr):
    """Process audio using overlapping windows and create 2D spectrograms"""
    # Calculate segment length and hop length in samples
    segment_length = int(Config.SEGMENT_DURATION * sr)
    hop_length = int(segment_length * (1 - Config.OVERLAP))
    
    # Calculate number of segments
    num_segments = 1 + (len(audio) - segment_length) // hop_length
    num_segments = min(num_segments, Config.MAX_SEGMENTS)
    
    # Extract spectrograms for each segment
    segment_specs = []
    
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
    
    # Stack segments into a single array
    # Result will have shape (MAX_SEGMENTS, n_mels, time_steps)
    return np.stack(segment_specs[:Config.MAX_SEGMENTS])

# Function to load and process data for CNN
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
            instrument_class = 'khaen'
        elif parent.startswith('khong-vong-') or parent.startswith('khong_vong_') or parent == 'khong_wong' or parent == 'khong_vong':
            instrument_class = 'khong_vong'
        elif parent.startswith('pin-') or parent.startswith('pin_') or parent == 'pin':
            instrument_class = 'pin'
        elif parent.startswith('ra-nad-') or parent.startswith('ra_nad_') or parent == 'ranad':
            instrument_class = 'ranad'
        elif parent.startswith('saw-') or parent.startswith('saw_') or parent == 'so_u':
            instrument_class = 'so_u'
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
                
                # Process audio for CNN
                specs = process_audio_for_cnn(audio, sr)
                features_list.append(specs)
                labels.append(instrument_class)
                file_paths.append(file_path)
                
                # Apply data augmentation
                augmented_samples = augment_audio(audio, sr)
                
                # Process augmented samples
                for aug_audio in augmented_samples:
                    aug_specs = process_audio_for_cnn(aug_audio, sr)
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

# Build CNN model with temporal attention - Intel optimized
def build_cnn_with_attention(input_shape, num_classes):
    """Build a CNN model with attention for audio classification"""
    # Input for segments - shape: (MAX_SEGMENTS, n_mels, time_steps)
    segments_input = tf.keras.layers.Input(shape=input_shape)
    
    # Build CNN feature extractor that will be applied to each segment
    def create_segment_encoder():
        inputs = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2]))
        
        # Add channel dimension for 2D convolutions
        x = tf.keras.layers.Reshape((input_shape[1], input_shape[2], 1))(inputs)
        
        # First convolutional block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Second convolutional block
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third convolutional block
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Global pooling to reduce dimensions
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    # Create the segment encoder
    segment_encoder = create_segment_encoder()
    
    # Apply the encoder to each segment using TimeDistributed
    encoded_segments = tf.keras.layers.TimeDistributed(segment_encoder)(segments_input)
    
    # Add attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(encoded_segments)
    attention = tf.keras.layers.Reshape((Config.MAX_SEGMENTS,))(attention)
    attention_weights = tf.keras.layers.Softmax()(attention)
    
    # Apply attention weights
    context = tf.keras.layers.Dot(axes=1)([attention_weights, encoded_segments])
    
    # Final classification layers
    x = tf.keras.layers.Dense(128, activation='relu')(context)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
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

# Export model to ONNX format (optional)
def export_to_onnx(model, filepath_base, input_names=None, output_names=None):
    """Export TensorFlow model to ONNX format"""
    try:
        import tf2onnx
        import onnx
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filepath_base)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Converting model to ONNX format...")
        
        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # Convert model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=None,  # Will be inferred
            opset=13,
            output_path=f"{filepath_base}.onnx",
            inputs_as_nchw=None
        )
        
        print(f"Model saved in ONNX format at: {filepath_base}.onnx")
        
        # Save model metadata for easier loading
        model_meta = {
            "input_shape": list(model.input_shape[1:]),  # Remove batch dimension
            "output_shape": list(model.output_shape[1:]),
            "classes": classes.tolist() if 'classes' in globals() else []
        }
        
        with open(f"{filepath_base}_meta.json", "w") as f:
            json.dump(model_meta, f)
        
        print(f"Model metadata saved at: {filepath_base}_meta.json")
    except ImportError:
        print("Warning: tf2onnx or onnx not installed. Skipping ONNX export.")
        print("To export to ONNX format, install tf2onnx: pip install tf2onnx onnx")

# Main function
def main():
    start_time = time.time()
    
    # Create necessary directories
    create_directory_if_not_exists("models")
    
    # Load and process data
    features_list, labels, classes, file_paths = load_data_for_cnn()
    
    # Convert string labels to numerical indices
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in labels])
    
    # Convert list of features to numpy array
    print("Converting features to numpy array...")
    X = np.array(features_list)
    
    # Get input shape for the model
    input_shape = X[0].shape
    
    # Split into training and testing sets (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert data to float32 for better compatibility
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print("\nBuilding CNN model with attention...")
    # Create and compile the model
    model = build_cnn_with_attention(input_shape, len(classes))
    
    # Use Intel optimizations if available
    if Config.INTEL_OPTIMIZED and 'itex' in globals():
        optimizer = tf.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Performance test
    print("\nPerforming performance test...")
    batch = X_train[:Config.BATCH_SIZE]
    
    # Warm-up run
    _ = model.predict(batch)
    
    # Timed run
    start_infer = time.time()
    _ = model.predict(batch)
    inference_time = time.time() - start_infer
    
    print(f"Inference time for {Config.BATCH_SIZE} samples: {inference_time:.4f} seconds")
    print(f"Inference speed: {Config.BATCH_SIZE/inference_time:.2f} samples/second")
    
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
            filepath=f"{Config.MODEL_SAVE_PATH}_checkpoint.h5",
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
    plt.savefig('training_history_cnn.png')
    print("Training history saved to training_history_cnn.png")
    
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
    plt.savefig('confusion_matrix_cnn.png')
    print("Confusion matrix saved to confusion_matrix_cnn.png")
    
    # Classification report
    print("\nClassification Report:")
    cr = classification_report(
        y_test, y_pred_classes, 
        target_names=classes
    )
    print(cr)
    
    # Save classification report to file
    with open('classification_report.txt', 'w') as f:
        f.write(cr)
    
    print("\nSaving model...")
    # Save standard model in TensorFlow format
    model.save(f'{Config.MODEL_SAVE_PATH}.h5')
    
    # Save TensorFlow SavedModel format (recommended)
    model.save(f'{Config.MODEL_SAVE_PATH}_saved_model')
    
    # Export to ONNX
    export_to_onnx(model, Config.MODEL_SAVE_PATH)
    
    # Save the class labels for easier loading
    with open('label_encoder.txt', 'w') as f:
        for label in classes:
            f.write(f"{label}\n")
    
    print("Model, ONNX export, and label encoder saved successfully!")
    
    # Print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print("\nTesting model on sample files...")
    # Test with a few samples
    test_samples = np.random.choice(file_paths, min(5, len(file_paths)), replace=False)
    
    # Function to predict instrument for a given audio file
    def predict_instrument(audio_path):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Process audio for CNN
            specs = process_audio_for_cnn(audio, sr)
            specs = np.expand_dims(specs, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(specs)[0]
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

if __name__ == "__main__":
    main()
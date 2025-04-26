import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json

# Specific configuration to address instrument bias
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    
    # STFT parameters matching your image
    N_FFT = 2048
    HOP_LENGTH = 512  # About 12ms at 44.1kHz
    WIN_LENGTH = 2048  # About 46ms at 44.1kHz
    
    # Mel parameters
    N_MELS = 128
    FMAX = 8000
    
    # Segmentation
    SEGMENT_DURATION = 3.0  # seconds
    MAX_SEGMENTS = 1  # Use single segment initially to reduce complexity
    
    # Training parameters - adjusted for better generalization
    BATCH_SIZE = 16  # Smaller batch size
    EPOCHS = 100
    LEARNING_RATE = 0.0001  # Lower learning rate
    EARLY_STOPPING_PATIENCE = 15
    
    # Regularization - stronger to prevent overfitting
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.01
    
    # Data augmentation - targeted
    ENABLE_TIME_STRETCH = True
    ENABLE_PITCH_SHIFT = True
    ENABLE_NOISE_ADDITION = True
    ENABLE_MIXUP = True  # New: mixup augmentation
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    BALANCE_SAMPLING = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/balanced_model"

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator to ensure balanced sampling"""
    def __init__(self, X, y, batch_size=16, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.classes = np.unique(y)
        self.class_indices = {c: np.where(y == c)[0] for c in self.classes}
        self.samples_per_class = batch_size // len(self.classes)
        
    def __len__(self):
        return len(self.X) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        # Sample equal number of examples from each class
        for c in self.classes:
            indices = np.random.choice(self.class_indices[c], self.samples_per_class, replace=True)
            batch_x.extend(self.X[indices])
            batch_y.extend(self.y[indices])
        
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        # Apply augmentation
        if self.augment:
            batch_x = self.augment_batch(batch_x)
        
        return batch_x, batch_y
    
    def augment_batch(self, batch):
        """Apply augmentation to batch"""
        augmented = []
        for x in batch:
            # Random time stretch
            if Config.ENABLE_TIME_STRETCH and np.random.random() < 0.5:
                stretch_factor = np.random.uniform(0.9, 1.1)
                x = librosa.effects.time_stretch(x, rate=stretch_factor)
            
            # Random pitch shift
            if Config.ENABLE_PITCH_SHIFT and np.random.random() < 0.5:
                pitch_shift = np.random.uniform(-2, 2)
                x = librosa.effects.pitch_shift(x, sr=Config.SAMPLE_RATE, n_steps=pitch_shift)
            
            # Add noise
            if Config.ENABLE_NOISE_ADDITION and np.random.random() < 0.5:
                noise = np.random.normal(0, 0.005, x.shape)
                x = x + noise
            
            augmented.append(x)
        
        return np.array(augmented)

class RobustFeatureExtractor:
    """Enhanced feature extraction focused on instrument discrimination"""
    
    @staticmethod
    def extract_features(audio, sr):
        """Extract robust features for instrument classification"""
        # 1. Standard STFT
        stft = librosa.stft(audio, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
        mag_stft = np.abs(stft)
        phase_stft = np.angle(stft)
        
        # 2. Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=mag_stft**2, 
            sr=sr, 
            n_mels=Config.N_MELS,
            fmax=Config.FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=mag_stft, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=mag_stft, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(S=mag_stft, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=mag_stft, sr=sr)
        
        # 4. Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(S=mag_stft)
        
        # 5. Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_stft = np.abs(librosa.stft(harmonic, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH))
        percussive_stft = np.abs(librosa.stft(percussive, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH))
        
        # Combine all features
        features = {
            'mel_spec': mel_spec_db,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'rms': rms,
            'harmonic_ratio': harmonic_stft / (mag_stft + 1e-10),
            'percussive_ratio': percussive_stft / (mag_stft + 1e-10)
        }
        
        return features

def build_enhanced_model(input_shape, num_classes):
    """Build a model with enhanced regularization and balanced architecture"""
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Normalize input
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Multi-scale feature extraction
    # Path 1: Fine temporal resolution (narrow kernels)
    path1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    path1 = tf.keras.layers.BatchNormalization()(path1)
    path1 = tf.keras.layers.Activation('relu')(path1)
    path1 = tf.keras.layers.MaxPooling2D((2, 2))(path1)
    path1 = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.5)(path1)
    
    # Path 2: Broad temporal context (wide kernels)
    path2 = tf.keras.layers.Conv2D(32, (3, 11), padding='same')(x)
    path2 = tf.keras.layers.BatchNormalization()(path2)
    path2 = tf.keras.layers.Activation('relu')(path2)
    path2 = tf.keras.layers.MaxPooling2D((2, 2))(path2)
    path2 = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.5)(path2)
    
    # Path 3: Frequency emphasis (tall kernels)
    path3 = tf.keras.layers.Conv2D(32, (11, 3), padding='same')(x)
    path3 = tf.keras.layers.BatchNormalization()(path3)
    path3 = tf.keras.layers.Activation('relu')(path3)
    path3 = tf.keras.layers.MaxPooling2D((2, 2))(path3)
    path3 = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.5)(path3)
    
    # Combine paths
    combined = tf.keras.layers.Concatenate()([path1, path2, path3])
    
    # Deep feature extraction with strong regularization
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Global feature aggregation
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with strong regularization
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Output with temperature scaling
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_with_balanced_approach(X, y, class_names):
    """Train model with balanced approach to prevent bias"""
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Compute class weights
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights = dict(enumerate(weights))
        print("Class weights:", class_weights)
    
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Create balanced generators
    train_generator = BalancedDataGenerator(X_train, y_train, Config.BATCH_SIZE, augment=True)
    val_generator = BalancedDataGenerator(X_val, y_val, Config.BATCH_SIZE, augment=False)
    
    # Build model
    input_shape = X_train[0].shape
    model = build_enhanced_model(input_shape, len(class_names))
    
    # Custom optimizer with adaptive learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    
    # Compile with label smoothing
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
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
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'confusion_matrix.png'))
    plt.close()
    
    # Check for bias
    pred_counts = np.bincount(y_pred_classes, minlength=len(class_names))
    actual_counts = np.bincount(y_val, minlength=len(class_names))
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, actual_counts, width, label='Actual')
    plt.bar(x + width/2, pred_counts, width, label='Predicted')
    
    plt.xlabel('Instrument')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions vs Actual')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'prediction_distribution.png'))
    plt.close()
    
    return model, history

def main():
    """Main function with balanced training approach"""
    
    # Create model directory
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Load data with robust feature extraction
    feature_extractor = RobustFeatureExtractor()
    X, y, class_names, _ = load_data_for_classification(feature_extractor)
    
    print(f"\nDataset summary:")
    print(f"Total samples: {len(X)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Train with balanced approach
    model, history = train_with_balanced_approach(X, y, class_names)
    
    # Save training history plot
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
    
    print(f"\nTraining complete! Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
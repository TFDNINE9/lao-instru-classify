import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import json
import random
from tqdm import tqdm



class OverfittingTester:
    def __init__(self, model_path, feature_extractor, class_labels):
        self.model = tf.keras.models.load_model(model_path)
        self.feature_extractor = feature_extractor
        self.class_labels = class_labels
        
    def test_on_new_recordings(self, test_folder):
        """Test model on completely new recordings"""
        print("\nTesting on new recordings...")
        results = []
        
        for file in os.listdir(test_folder):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(test_folder, file)
                try:
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=44100)
                    
                    # Extract features
                    segments = process_audio_segments(audio, sr, self.feature_extractor)
                    segments = np.expand_dims(segments, 0)  # Add batch dimension
                    
                    # Predict
                    prediction = self.model.predict(segments)[0]
                    predicted_class = self.class_labels[np.argmax(prediction)]
                    confidence = np.max(prediction)
                    
                    results.append({
                        'file': file,
                        'predicted': predicted_class,
                        'confidence': confidence,
                        'all_predictions': {self.class_labels[i]: float(prediction[i]) 
                                          for i in range(len(self.class_labels))}
                    })
                    
                    print(f"{file}: {predicted_class} ({confidence:.2f})")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        return results
    
    def dropout_analysis(self, X_test, y_test):
        """Compare model performance with and without dropout"""
        print("\nPerforming dropout analysis...")
        
        # Get original performance
        orig_pred = self.model.predict(X_test)
        orig_acc = np.mean(np.argmax(orig_pred, axis=1) == y_test)
        
        # Create model copy without dropout
        no_dropout_model = tf.keras.models.clone_model(self.model)
        
        # Remove dropout layers
        for layer in no_dropout_model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = 0.0
        
        # Compile and evaluate
        no_dropout_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        no_dropout_model.set_weights(self.model.get_weights())
        
        no_dropout_pred = no_dropout_model.predict(X_test)
        no_dropout_acc = np.mean(np.argmax(no_dropout_pred, axis=1) == y_test)
        
        print(f"Original accuracy: {orig_acc:.4f}")
        print(f"Without dropout: {no_dropout_acc:.4f}")
        
        return orig_acc, no_dropout_acc
    
    def learning_curve_analysis(self, X, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """Generate learning curves to check for overfitting"""
        print("\nGenerating learning curves...")
        
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            scores = []
            for i in range(3):  # 3 runs for averaging
                # Split data
                split_idx = int(len(X) * train_size)
                indices = np.random.permutation(len(X))
                train_idx = indices[:split_idx]
                val_idx = indices[split_idx:]
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train a new model
                model = tf.keras.models.clone_model(self.model)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                history = model.fit(X_train, y_train, 
                                  validation_data=(X_val, y_val),
                                  epochs=20, batch_size=32, verbose=0)
                
                scores.append([history.history['accuracy'][-1], history.history['val_accuracy'][-1]])
            
            scores = np.array(scores)
            train_scores.append(np.mean(scores[:, 0]))
            val_scores.append(np.mean(scores[:, 1]))
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score')
        plt.plot(train_sizes, val_scores, 'o-', label='Validation score')
        plt.xlabel('Training set size (fraction)')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png')
        plt.close()
        
        return train_scores, val_scores
    
    def noise_robustness_test(self, X_test, y_test, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """Test model robustness to noise"""
        print("\nTesting noise robustness...")
        
        original_acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
        noise_accuracies = []
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            
            # Evaluate
            acc = self.model.evaluate(X_noisy, y_test, verbose=0)[1]
            noise_accuracies.append(acc)
            
            print(f"Noise level {noise_level}: Accuracy {acc:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot([0] + noise_levels, [original_acc] + noise_accuracies, 'o-')
        plt.xlabel('Noise level')
        plt.ylabel('Accuracy')
        plt.title('Model Robustness to Noise')
        plt.grid(True)
        plt.savefig('noise_robustness.png')
        plt.close()
        
        return noise_accuracies
    
    def confidence_analysis(self, X_test, y_test):
        """Analyze prediction confidence for correct vs incorrect predictions"""
        print("\nAnalyzing prediction confidence...")
        
        predictions = self.model.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        correct_mask = (pred_classes == y_test)
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Plot confidence distributions
        plt.figure(figsize=(10, 6))
        plt.hist(correct_confidences, bins=20, alpha=0.5, label='Correct', density=True)
        plt.hist(incorrect_confidences, bins=20, alpha=0.5, label='Incorrect', density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution for Predictions')
        plt.legend()
        plt.grid(True)
        plt.savefig('confidence_distribution.png')
        plt.close()
        
        print(f"Average confidence for correct predictions: {np.mean(correct_confidences):.4f}")
        print(f"Average confidence for incorrect predictions: {np.mean(incorrect_confidences):.4f}")
        
        return correct_confidences, incorrect_confidences
    
    def instrument_bias_test(self, X_test, y_test):
        """Test if model is biased towards certain instruments"""
        print("\nTesting for instrument bias...")
        
        predictions = self.model.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Count predictions for each class
        pred_counts = {}
        actual_counts = {}
        
        for label in self.class_labels:
            pred_counts[label] = 0
            actual_counts[label] = 0
        
        for pred, actual in zip(pred_classes, y_test):
            pred_label = self.class_labels[pred]
            actual_label = self.class_labels[actual]
            pred_counts[pred_label] += 1
            actual_counts[actual_label] += 1
        
        # Calculate bias metrics
        total_samples = len(y_test)
        bias_metrics = {}
        
        for label in self.class_labels:
            expected_ratio = actual_counts[label] / total_samples
            predicted_ratio = pred_counts[label] / total_samples
            bias = predicted_ratio - expected_ratio
            bias_metrics[label] = {
                'expected': expected_ratio,
                'predicted': predicted_ratio,
                'bias': bias
            }
        
        # Plot bias
        labels = list(bias_metrics.keys())
        biases = [bias_metrics[label]['bias'] for label in labels]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, biases)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.ylabel('Prediction Bias')
        plt.title('Model Bias Towards Instruments')
        plt.xticks(rotation=45)
        
        # Color bars based on bias
        for bar, bias in zip(bars, biases):
            if abs(bias) > 0.1:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        plt.tight_layout()
        plt.savefig('instrument_bias.png')
        plt.close()
        
        # Print bias summary
        print("\nBias Summary:")
        for label, metrics in bias_metrics.items():
            bias_percentage = metrics['bias'] * 100
            if abs(bias_percentage) > 10:
                print(f"WARNING: {label} shows significant bias: {bias_percentage:.1f}%")
            else:
                print(f"{label}: {bias_percentage:.1f}% bias")
        
        return bias_metrics

def main():
    """Main function to run all overfitting tests"""
    # Load your trained model and test data
    model_path = 'models/lao_instruments_model_optimized_best.h5'  # Adjust path as needed
    feature_extractor = FeatureExtractor(method='mel')  # Use the same feature extractor as training
    
    # Load class labels
    class_labels = ['background', 'khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']
    
    # Load test data (you should have a separate test set)
    X_test, y_test, _, _ = load_data_for_classification(feature_extractor)
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_labels)}
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Initialize tester
    tester = OverfittingTester(model_path, feature_extractor, class_labels)
    
    # Run all tests
    print("Running comprehensive overfitting analysis...")
    
    # 1. Test on completely new recordings (if you have a test folder)
    if os.path.exists('test_recordings'):
        test_results = tester.test_on_new_recordings('test_recordings')
        
        # Save results
        with open('new_recordings_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
    
    # 2. Dropout analysis
    orig_acc, no_dropout_acc = tester.dropout_analysis(X_test, y_test_encoded)
    
    # 3. Learning curve analysis
    train_scores, val_scores = tester.learning_curve_analysis(X_test, y_test_encoded)
    
    # 4. Noise robustness test
    noise_accuracies = tester.noise_robustness_test(X_test, y_test_encoded)
    
    # 5. Confidence analysis
    correct_conf, incorrect_conf = tester.confidence_analysis(X_test, y_test_encoded)
    
    # 6. Instrument bias test
    bias_metrics = tester.instrument_bias_test(X_test, y_test_encoded)
    
    # Generate comprehensive report
    report = {
        'dropout_analysis': {
            'original_accuracy': float(orig_acc),
            'no_dropout_accuracy': float(no_dropout_acc),
            'dropout_effectiveness': float(orig_acc - no_dropout_acc)
        },
        'learning_curve': {
            'train_scores': [float(s) for s in train_scores],
            'val_scores': [float(s) for s in val_scores],
            'overfitting_detected': bool(train_scores[-1] - val_scores[-1] > 0.1)
        },
        'noise_robustness': {
            'noise_levels': [0.01, 0.05, 0.1, 0.2],
            'accuracies': [float(acc) for acc in noise_accuracies],
            'degradation': float(orig_acc - noise_accuracies[-1])
        },
        'confidence_analysis': {
            'correct_avg_confidence': float(np.mean(correct_conf)),
            'incorrect_avg_confidence': float(np.mean(incorrect_conf)),
            'confidence_gap': float(np.mean(correct_conf) - np.mean(incorrect_conf))
        },
        'instrument_bias': bias_metrics
    }
    
    # Save report
    with open('overfitting_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("="*50)
    
    if report['learning_curve']['overfitting_detected']:
        print("⚠️ WARNING: Overfitting detected in learning curves")
    else:
        print("✅ No significant overfitting detected in learning curves")
    
    if abs(report['dropout_analysis']['dropout_effectiveness']) > 0.05:
        print("⚠️ WARNING: Large difference with/without dropout suggests overfitting")
    else:
        print("✅ Dropout effectiveness is within acceptable range")
    
    if report['noise_robustness']['degradation'] > 0.2:
        print("⚠️ WARNING: Poor noise robustness suggests over-specialization")
    else:
        print("✅ Model shows reasonable noise robustness")
    
    if report['confidence_analysis']['confidence_gap'] < 0.1:
        print("⚠️ WARNING: Low confidence gap suggests overconfident predictions")
    else:
        print("✅ Model shows appropriate confidence differentiation")
    
    # Check for significant bias
    bias_issues = False
    for label, metrics in bias_metrics.items():
        if abs(metrics['bias']) > 0.1:
            bias_issues = True
            print(f"⚠️ WARNING: Significant bias towards {label}")
    
    if not bias_issues:
        print("✅ No significant instrument bias detected")
    
    print("\nDetailed report saved to 'overfitting_analysis_report.json'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()
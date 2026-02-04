"""
AUDIO CLASSIFICATION USING K-NEAREST NEIGHBORS (KNN)

This program uses time domain features to classify WAV files using a K-Nearest Neighbours (KNN) algorithm. 
It extracts mean amplitude, standard deviation, Root Mean Square (RMS) energy and Zero-Crossing Rate (ZCR)
from the WAV files, and uses them to predict which class a new audio file belongs to.

PARAMETERS:
- train_ratio = Proportion of data used for training (default: 0.8 = 80%)
- k = Number of nearest neighbors to consider (default: 5)
- sr = Sample rate for loading audio (default: 22050 Hz)
- random seed = For reproducibility (default: 42)

"""

import os
import numpy as np
import librosa
from collections import Counter

# Extract time domain features
def extract_features(filepath, sr=22050):
    """Extract audio features from a WAV file."""
    try:
        y, sr = librosa.load(filepath, sr=sr, mono=True)
        features = [
            np.mean(y),
            np.std(y),
            np.sqrt(np.mean(y ** 2)),                     # RMS
            np.mean(librosa.feature.zero_crossing_rate(y))
        ]
        return np.array(features)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load dataset with shuffling
def load_dataset(base_path):
    """Load dataset and return features and labels."""
    X = []
    y = []
    
    for label_name in sorted(os.listdir(base_path)):
        label_path = os.path.join(base_path, label_name)
        if not os.path.isdir(label_path):
            continue
        
        label = label_name  # Use the Folder name as label
        files_loaded = 0
        
        for file in sorted(os.listdir(label_path)):
            if file.endswith(".wav"):
                filepath = os.path.join(label_path, file)
                features = extract_features(filepath)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    files_loaded += 1
        
        print(f"Loaded {files_loaded} files for class '{label_name}'")
    
    return np.array(X), np.array(y)

# Normalize features
def normalize(X):
    """Normalize features to zero mean and unit variance."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

# Implement K-nearest neighbours
def knn_predict(X_train, y_train, x_test, k=5):
    """Predict class label using K-nearest neighbors."""
    # Euclidean distance
    distances = np.linalg.norm(X_train - x_test, axis=1)
    # Indices of k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    # Get their labels
    k_labels = y_train[k_indices]
    # Majority vote
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Stratified train/test split
def stratified_split(X, y, train_ratio=0.8):
    """
    Split data while maintaining class proportions.
    With 20 samples per class: 16 train, 4 test per class.
    """
    unique_labels = np.unique(y)
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for label in unique_labels:
        # Get indices for this class
        label_indices = np.where(y == label)[0]
        
        # Shuffle indices for this class
        np.random.shuffle(label_indices)
        
        # Split indices
        split_idx = int(len(label_indices) * train_ratio)
        train_indices = label_indices[:split_idx]
        test_indices = label_indices[split_idx:]
        
        # Append to lists
        X_train_list.append(X[train_indices])
        X_test_list.append(X[test_indices])
        y_train_list.append(y[train_indices])
        y_test_list.append(y[test_indices])
    
    # Concatenate all classes
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    
    # Shuffle the combined training and test sets
    train_shuffle = np.random.permutation(len(X_train))
    test_shuffle = np.random.permutation(len(X_test))
    
    return X_train[train_shuffle], X_test[test_shuffle], y_train[train_shuffle], y_test[test_shuffle]

# Evaluate model
def evaluate_model(X_train, y_train, X_test, y_test, k=5):
    """Evaluate KNN model and return detailed results."""
    predictions = []
    correct = 0
    
    for x, label in zip(X_test, y_test):
        pred = knn_predict(X_train, y_train, x, k=k)
        predictions.append(pred)
        if pred == label:
            correct += 1
    
    accuracy = correct / len(X_test)
    
    # Per-class accuracy
    unique_labels = np.unique(y_test)
    print("\nPer-class accuracy:")
    for label in unique_labels:
        label_mask = y_test == label
        label_correct = sum((np.array(predictions)[label_mask] == y_test[label_mask]))
        label_total = sum(label_mask)
        label_accuracy = label_correct / label_total if label_total > 0 else 0
        print(f"  {label}: {label_accuracy:.2%} ({label_correct}/{label_total})")
    
    return accuracy, predictions

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load dataset
    print("Loading dataset...")
    data_path = "data"
    
    if not os.path.exists(data_path):
        print(f"Error: Directory '{data_path}' not found!")
        exit(1)
    
    X, y = load_dataset(data_path)
    
    # Validate dataset
    if len(X) == 0:
        print("Error: No audio files found in the dataset!")
        exit(1)
    
    print(f"\nTotal samples loaded: {len(X)}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    # Normalize features
    X_normalized, mean, std = normalize(X)
    
    # Stratified train/test split (16 train, 4 test per class)
    X_train, X_test, y_train, y_test = stratified_split(X_normalized, y, train_ratio=0.8)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Test different k values
    print("\n" + "="*50)
    print("Testing different k values:")
    print("="*50)
    
    for k in [1, 3, 5, 7]:
        print(f"\nk = {k}:")
        accuracy, predictions = evaluate_model(X_train, y_train, X_test, y_test, k=k)
        print(f"Overall Accuracy: {accuracy:.2%}")
    
    # Use best k value (you can adjust based on results)
    best_k = 5
    print("\n" + "="*50)
    print(f"Final model with k={best_k}")
    print("="*50)
    

# Test with a file from your dataset
print("\n" + "="*50)
print("Testing with existing dataset files:")
print("="*50)

# Test a piano file
test_classes = ["piano", "snare"]
for test_class in test_classes:
    class_path = os.path.join("data", test_class)
    if os.path.exists(class_path):
        test_files = [f for f in os.listdir(class_path) if f.endswith(".wav")]
        if test_files:
            # Test with first file
            test_file = os.path.join(class_path, test_files[0])
            print(f"\nTesting {test_class} file: {test_files[0]}")
            features = extract_features(test_file)
            if features is not None:
                features_normalized = (features - mean) / std
                prediction = knn_predict(X_train, y_train, features_normalized, k=best_k)
                print(f"  Actual class: {test_class}")
                print(f"  Predicted class: {prediction}")
                print(f"  Correct: {'Yes' if prediction == test_class else 'No'}")

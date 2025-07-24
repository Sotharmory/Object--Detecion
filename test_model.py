import numpy as np
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from main import FruitDetectorML, process_xml, extract_features

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir="evaluation_results"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_class_distribution(y_true, classes, model_name, save_dir="evaluation_results"):
    """
    Plot class distribution in test set
    """
    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(y_true)
    plt.bar(classes, class_counts)
    plt.title(f'Class Distribution in Test Set - {model_name}')
    plt.ylabel('Number of Samples')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'class_distribution_{model_name}.png'))
    plt.close()

def test_model(model_path, test_path):
    """
    Test a trained model on the test dataset
    
    Args:
        model_path (str): Path to the saved model file (.pkl)
        test_path (str): Path to the test dataset directory
    """
    # Load the model
    detector = FruitDetectorML()
    detector.load_model(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Process test data
    print(f"\nTesting model: {model_name}")
    print("Processing test data...")
    
    # Get test features and labels
    test_features = []
    test_labels = []
    processed_images = []
    
    image_files = glob.glob(os.path.join(test_path, "*.jpg"))
    
    for img_file in image_files:
        # Read image
        image = cv2.imread(img_file)
        if image is None:
            continue
            
        # Find corresponding XML file
        xml_file = img_file.replace('.jpg', '.xml')
        if not os.path.exists(xml_file):
            continue
        
        try:
            # Process annotation
            annotation = process_xml(xml_file)
            if not annotation:
                continue
                
            label = annotation[0][-1][0]  # Get label
            bbox = annotation[0][0]  # Get bounding box
            
            # Crop object from bounding box
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            # Extract features using the imported function
            feature_vector = extract_features(cropped)
            test_features.append(feature_vector)
            test_labels.append(detector.label_dict[label])
            processed_images.append(img_file)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    if len(test_features) == 0:
        print("No valid test samples were processed!")
        return None, None
        
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    # Scale features
    test_features_scaled = detector.scaler.transform(test_features)
    
    # Make predictions
    predictions = detector.classifier.predict(test_features_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    # Get class names
    class_names = [detector.reverse_label_dict[i] for i in sorted(detector.reverse_label_dict.keys())]
    
    # Generate and save evaluation plots
    plot_confusion_matrix(test_labels, predictions, class_names, model_name)
    plot_class_distribution(test_labels, class_names, model_name)
    
    # Generate classification report
    report = classification_report(test_labels, predictions, target_names=class_names)
    
    # Save classification report
    save_dir = "evaluation_results"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print(f"\nEvaluation results saved in {save_dir}/")
    
    return accuracy, report

def test_single_image(model_path, image_path, threshold=0.7):
    """
    Test a trained model on a single image
    
    Args:
        model_path (str): Path to the saved model file (.pkl)
        image_path (str): Path to the test image
        threshold (float): Confidence threshold for detection
    """
    # Load the model
    detector = FruitDetectorML()
    detector.load_model(model_path)
    
    # Detect object
    result = detector.detect_object(image_path, threshold=threshold)
    
    if result:
        print(f"\nDetected: {result['label']} with confidence {result['confidence']:.2f}")
        detector.visualize_detection(image_path, result)
    else:
        print("No object detected")
    
    return result

if __name__ == "__main__":
    # Paths
    test_path = "archive/test_zip/test/"
    rf_model_path = "fruit_detector_rf.pkl"
    svm_model_path = "fruit_detector_svm.pkl"
    
    # Test Random Forest model
    print("\n=== Testing Random Forest Model ===")
    rf_accuracy, rf_report = test_model(rf_model_path, test_path)
    
    # Test SVM model
    print("\n=== Testing SVM Model ===")
    svm_accuracy, svm_report = test_model(svm_model_path, test_path)
    
    # Compare models
    if rf_accuracy is not None and svm_accuracy is not None:
        print("\n=== Model Comparison ===")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        
        # Plot accuracy comparison
        plt.figure(figsize=(8, 6))
        models = ['Random Forest', 'SVM']
        accuracies = [rf_accuracy, svm_accuracy]
        plt.bar(models, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Save plot
        save_dir = "evaluation_results"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close()

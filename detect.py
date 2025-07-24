import numpy as np
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from main import FruitDetectorML
import time

def load_random_test_image(test_path):
    """
    Load one random image from test dataset
    """
    image_files = glob.glob(os.path.join(test_path, "*.jpg"))
    
    # Randomly select one image
    selected_image = random.choice(image_files)
    return selected_image

def detect_and_visualize(model_path, image_path, threshold=0.5, save_path=None):
    """
    Detect objects in image and visualize results
    """
    try:
        start_time = time.time()
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\nProcessing with {model_name}...")
        
        # Load model
        detector = FruitDetectorML()
        detector.load_model(model_path)
        print("Model loaded successfully")
        
        # Read image
        print("Reading image...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Detect objects
        print("Detecting objects...")
        result = detector.detect_object(image_path, threshold=threshold)
        
        if result:
            # Get detection results
            label = result['label']
            confidence = result['confidence']
            bbox = result['bbox']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and confidence
            plt.text(
                x1, y1-10,
                f'{label}: {confidence:.2f}',
                bbox=dict(facecolor='yellow', alpha=0.7),
                fontsize=12
            )
            
            plt.title(f"{model_name} - Detected: {label} (Confidence: {confidence:.2f})")
        else:
            plt.title(f"{model_name} - No object detected")
        
        plt.axis('off')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing image with {model_name}: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def test_with_both_models(rf_model_path, svm_model_path, test_path, threshold=0.5):
    """
    Test one image with both models
    """
    try:
        # Create output directory for results
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load one random test image
        print("\nLoading random test image...")
        test_image = load_random_test_image(test_path)
        print(f"Selected image: {os.path.basename(test_image)}")
        
        # Test with both models
        models = [rf_model_path, svm_model_path]
        
        for model_path in models:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Generate output path
            output_path = os.path.join(
                output_dir,
                f"result_{model_name}_{os.path.splitext(os.path.basename(test_image))[0]}.png"
            )
            
            # Detect and save results
            detect_and_visualize(
                model_path,
                test_image,
                threshold=threshold,
                save_path=output_path
            )
            
    except Exception as e:
        print(f"Error in test_with_both_models: {str(e)}")

if __name__ == "__main__":
    # Paths
    test_path = "archive/test_zip/test/"
    rf_model_path = "fruit_detector_rf.pkl"
    svm_model_path = "fruit_detector_svm.pkl"
    
    # Test one random image with both models
    test_with_both_models(
        rf_model_path=rf_model_path,
        svm_model_path=svm_model_path,
        test_path=test_path,
        threshold=0.5  # Detection confidence threshold
    ) 

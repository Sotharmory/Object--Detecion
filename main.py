import numpy as np
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
np.random.seed(42)

def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
              break
        index += 1
    return index

def GetInt(name, root, index=0):
    return int(GetItem(name, root, index))

def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    return -1

def process_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    
    num_boxes = FindNumberBoundingBoxes(root)
    attributes = []
    
    for index in range(num_boxes):
        xmin = GetInt('xmin', root, index)
        ymin = GetInt('ymin', root, index)
        xmax = GetInt('xmax', root, index)
        ymax = GetInt('ymax', root, index)
        
        width = GetInt('width', root)
        height = GetInt('height', root)
        filename = GetItem('filename', root) + '.JPEG'
        label = GetItem('name', root)
        
        attributes.append([[xmin,ymin,xmax,ymax],[width],[height],[filename],[label]])
    return attributes

# Hàm trích xuất features từ ảnh
def extract_features(image):
    """
    Trích xuất các features từ ảnh để sử dụng cho ML truyền thống
    """
    # Resize ảnh về kích thước cố định
    resized = cv2.resize(image, (224, 224))
    
    # 1. Color features - Histogram
    hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
    color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
    
    # 2. Texture features - LBP (Local Binary Pattern)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Simplified LBP
    def lbp_simple(img):
        h, w = img.shape
        lbp = np.zeros((h-2, w-2))
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                code = 0
                code |= (img[i-1, j-1] >= center) << 7
                code |= (img[i-1, j] >= center) << 6
                code |= (img[i-1, j+1] >= center) << 5
                code |= (img[i, j+1] >= center) << 4
                code |= (img[i+1, j+1] >= center) << 3
                code |= (img[i+1, j] >= center) << 2
                code |= (img[i+1, j-1] >= center) << 1
                code |= (img[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        return lbp
    
    lbp = lbp_simple(gray)
    lbp_hist = np.histogram(lbp, bins=256, range=(0, 256))[0]
    
    # 3. Edge features - Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 4. Shape features - Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        shape_features = [area, perimeter, circularity, cx, cy]
    else:
        shape_features = [0, 0, 0, 0, 0]
    
    # 5. Statistical features
    mean_bgr = np.mean(resized, axis=(0, 1))
    std_bgr = np.std(resized, axis=(0, 1))
    
    # Kết hợp tất cả features
    all_features = np.concatenate([
        color_features,
        lbp_hist,
        [edge_density],
        shape_features,
        mean_bgr,
        std_bgr
    ])
    
    return all_features

# Hàm sliding window để tìm object
def sliding_window(image, window_size=(64, 64), step_size=32):
    """
    Sliding window approach để tìm object trong ảnh
    """
    windows = []
    h, w = image.shape[:2]
    
    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            windows.append((window, (x, y, x + window_size[0], y + window_size[1])))
    
    return windows

def plot_feature_importance(classifier, feature_names, save_dir="training_results"):
    """
    Plot feature importance for Random Forest classifier
    """
    if hasattr(classifier, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'feature_importances.png'))
        plt.close()

def plot_data_distribution(labels, label_dict, save_dir="training_results"):
    """
    Plot training data distribution
    """
    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(labels)
    classes = [label_dict[i] for i in range(len(class_counts))]
    
    plt.bar(classes, class_counts)
    plt.title('Training Data Distribution')
    plt.ylabel('Number of Samples')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_data_distribution.png'))
    plt.close()

# Class chính cho detection
class FruitDetectorML:
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_dict = {"apple": 0, "banana": 1, "orange": 2}
        self.reverse_label_dict = {0: "apple", 1: "banana", 2: "orange"}
        
    def prepare_data(self, train_path, val_path):
        """
        Chuẩn bị dữ liệu training và validation
        """
        print("Preparing training data...")
        train_features, train_labels = self._process_dataset(train_path)
        
        print("Preparing validation data...")
        val_features, val_labels = self._process_dataset(val_path)
        
        # Chuẩn hóa features
        all_features = np.vstack([train_features, val_features])
        self.scaler.fit(all_features)
        
        train_features_scaled = self.scaler.transform(train_features)
        val_features_scaled = self.scaler.transform(val_features)
        
        return train_features_scaled, train_labels, val_features_scaled, val_labels
    
    def _process_dataset(self, dataset_path):
        """
        Xử lý dataset và trích xuất features
        """
        features = []
        labels = []
        
        image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
        
        for img_file in image_files:
            # Đọc ảnh
            image = cv2.imread(img_file)
            if image is None:
                continue
                
            # Tìm file XML tương ứng
            xml_file = img_file.replace('.jpg', '.xml')
            if not os.path.exists(xml_file):
                continue
            
            try:
                # Xử lý annotation
                annotation = process_xml(xml_file)
                if not annotation:
                    continue
                    
                label = annotation[0][-1][0]  # Lấy label
                bbox = annotation[0][0]  # Lấy bounding box
                
                # Crop object từ bounding box
                x1, y1, x2, y2 = bbox
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size == 0:
                    continue
                
                # Trích xuất features
                feature_vector = extract_features(cropped)
                features.append(feature_vector)
                labels.append(self.label_dict[label])
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Training với Random Forest
        """
        print("Training Random Forest...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.classifier.predict(X_train))
        val_acc = accuracy_score(y_val, self.classifier.predict(X_val))
        
        print(f"Random Forest - Train Accuracy: {train_acc:.4f}")
        print(f"Random Forest - Validation Accuracy: {val_acc:.4f}")
        
        # Plot feature importance
        feature_names = [
            *[f'color_hist_{i}' for i in range(96)],  # 32 bins * 3 channels
            *[f'lbp_hist_{i}' for i in range(256)],
            'edge_density',
            *[f'shape_{i}' for i in range(5)],
            *[f'mean_bgr_{i}' for i in range(3)],
            *[f'std_bgr_{i}' for i in range(3)]
        ]
        plot_feature_importance(self.classifier, feature_names)
        
        return self.classifier
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """
        Training với SVM
        """
        print("Training SVM...")
        self.classifier = SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.classifier.predict(X_train))
        val_acc = accuracy_score(y_val, self.classifier.predict(X_val))
        
        print(f"SVM - Train Accuracy: {train_acc:.4f}")
        print(f"SVM - Validation Accuracy: {val_acc:.4f}")
        
        return self.classifier
    
    def detect_object(self, image_path, window_size=(64, 64), step_size=32, threshold=0.7):
        """
        Detect object trong ảnh sử dụng sliding window
        """
        if self.classifier is None:
            print("Model chưa được train!")
            return None
            
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print("Không thể đọc ảnh!")
            return None
        
        # Sliding window
        windows = sliding_window(image, window_size, step_size)
        
        best_score = 0
        best_prediction = None
        best_bbox = None
        
        for window, bbox in windows:
            if window.size == 0:
                continue
                
            # Trích xuất features
            features = extract_features(window)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(features_scaled)[0]
                max_proba = np.max(proba)
                
                if max_proba > best_score and max_proba > threshold:
                    best_score = max_proba
                    best_prediction = np.argmax(proba)
                    best_bbox = bbox
        
        if best_prediction is not None:
            return {
                'label': self.reverse_label_dict[best_prediction],
                'confidence': best_score,
                'bbox': best_bbox
            }
        
        return None
    
    def visualize_detection(self, image_path, detection_result):
        """
        Visualize kết quả detection
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image_rgb)
        
        if detection_result:
            label = detection_result['label']
            confidence = detection_result['confidence']
            bbox = detection_result['bbox']
            
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x1, y1-10, f'{label}: {confidence:.2f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_title(f"Detected: {label} (Confidence: {confidence:.2f})")
        else:
            ax.set_title("No object detected")
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename):
        """
        Lưu model
        """
        joblib.dump({
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_dict': self.label_dict,
            'reverse_label_dict': self.reverse_label_dict
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load model
        """
        model_data = joblib.load(filename)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_dict = model_data['label_dict']
        self.reverse_label_dict = model_data['reverse_label_dict']
        print(f"Model loaded from {filename}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo detector
    detector = FruitDetectorML()
    
    # Đường dẫn dữ liệu (cập nhật theo đường dẫn thực tế)
    train_path = "archive/train_zip/train/"
    val_path = "archive/test_zip/test/"
    
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    X_train, y_train, X_val, y_val = detector.prepare_data(train_path, val_path)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Plot training data distribution
    plot_data_distribution(y_train, detector.reverse_label_dict)
    
    # Training với Random Forest
    print("\n=== Training Random Forest ===")
    rf_model = detector.train_random_forest(X_train, y_train, X_val, y_val)
    
    # Lưu model Random Forest
    detector.save_model("fruit_detector_rf.pkl")
    
    # Training với SVM
    print("\n=== Training SVM ===")
    svm_model = detector.train_svm(X_train, y_train, X_val, y_val)
    
    # Lưu model SVM
    detector.save_model("fruit_detector_svm.pkl")
    
    # Test detection trên ảnh mẫu
    print("\n=== Testing Detection ===")
    test_image = "path/to/test/image.jpg"  # Cập nhật đường dẫn thực tế
    
    # Load model để test
    detector.load_model("fruit_detector_rf.pkl")
    
    # Detect object
    result = detector.detect_object(test_image)
    
    if result:
        print(f"Detected: {result['label']} with confidence {result['confidence']:.2f}")
        detector.visualize_detection(test_image, result)
    else:
        print("No object detected")
